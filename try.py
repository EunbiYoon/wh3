import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ===== Main Function =====
def main():
    data = load_and_preprocess_data("raisin.csv")
    fold_data = cross_validation(data, k_fold=5)

    ntrees_list, metrics = evaluate_random_forest(fold_data, k_fold=5)
    plot_metrics(ntrees_list, metrics, save_dir="raisin_result")
    
# ===== Preprocessing =====
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data = data.rename(columns={"class": "label"})

    for col in data.columns:
        if col == "label":
            continue
        elif col.endswith("_cat"):
            data[col] = data[col].astype(str)
        elif col.endswith("_num"):
            data[col] = (data[col] > data[col].mean()).astype(int)
        else:
            print("There is an error in csv file column name")
    return data

# ===== Cross-validation =====
def cross_validation(data, k_fold):
    class_0 = data[data['label'] == 0].reset_index(drop=True)
    class_1 = data[data['label'] == 1].reset_index(drop=True)

    all_data = pd.DataFrame()  # ❗ 하나의 DataFrame만 사용

    for i in range(k_fold):
        class_0_start = int(len(class_0) * i / k_fold)
        class_0_end = int(len(class_0) * (i + 1) / k_fold)
        class_1_start = int(len(class_1) * i / k_fold)
        class_1_end = int(len(class_1) * (i + 1) / k_fold)

        class_0_fold = class_0.iloc[class_0_start:class_0_end]
        class_1_fold = class_1.iloc[class_1_start:class_1_end]

        fold_data = pd.concat([class_0_fold, class_1_fold]).copy()
        fold_data["k_fold"] = i

        all_data = pd.concat([all_data, fold_data], ignore_index=True)  # 누적
    all_data.to_excel('aa.xlsx')
    return all_data

# ===== Decision Tree =====
def entropy(y):
    class_counts = y.value_counts()
    probabilities = class_counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values = X[feature].unique()
    weighted_entropy = sum(
        (len(y[X[feature] == value]) / len(y)) * entropy(y[X[feature] == value]) for value in values
    )
    return total_entropy - weighted_entropy

class Node:
    def __init__(self, feature=None, label=None, children=None):
        self.feature = feature
        self.label = label
        self.children = children if children else {}

def build_tree(X, y, features, depth=0, max_depth=5, min_info_gain=1e-5):
    if len(y.unique()) == 1 or len(features) == 0 or depth == max_depth:
        return Node(label=y.mode()[0])

    info_gains = {f: information_gain(X, y, f) for f in features}
    best_feature = max(info_gains, key=info_gains.get)
    best_gain = info_gains[best_feature]

    if best_gain < min_info_gain:
        return Node(label=y.mode()[0])

    tree = Node(feature=best_feature)
    remaining_features = [f for f in features if f != best_feature]

    for value in X[best_feature].unique():
        subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
        subset_y = y[X[best_feature] == value]
        tree.children[value] = build_tree(
            subset_X, subset_y, remaining_features, depth + 1, max_depth, min_info_gain
        )

    return tree

def tree_to_dict(node):
    if node.label is not None:
        return {"label": int(node.label)}  # ⭐ int64 → int 변환
    return {
        "feature": node.feature,
        "children": {str(k): tree_to_dict(v) for k, v in node.children.items()}
    }


def save_trees_as_json(trees, ntrees, base_dir="saved_trees"):
    folder = os.path.join(base_dir, f"ntrees_{ntrees}")
    os.makedirs(folder, exist_ok=True)
    for i, tree in enumerate(trees, start=1):
        tree_dict = tree_to_dict(tree)
        with open(os.path.join(folder, f"tree_{i}.json"), "w") as f:
            json.dump(tree_dict, f, indent=4)

def predict(tree, X):
    predictions = []
    for _, row in X.iterrows():
        node = tree
        while node.label is None:
            if row[node.feature] not in node.children:
                predictions.append(None)
                break
            node = node.children[row[node.feature]]
        else:
            predictions.append(node.label)
    return np.array(predictions)

def accuracy(predictions, true_labels):
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    valid = predictions != None
    return np.mean(predictions[valid] == true_labels[valid])

def precision(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score_manual(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

# ===== Random Forest Helpers =====
def bootstrap_sample(X, y):
    idxs = np.random.choice(len(X), size=len(X), replace=True)
    return X.iloc[idxs].reset_index(drop=True), y.iloc[idxs].reset_index(drop=True)

def random_forest_predict(trees, X):
    tree_preds = np.array([predict(tree, X) for tree in trees])
    final_preds = []
    for i in range(X.shape[0]):
        row_preds = tree_preds[:, i]
        values, counts = np.unique(row_preds[row_preds != None], return_counts=True)
        if len(counts) == 0:
            final_preds.append(None)
        else:
            final_preds.append(values[np.argmax(counts)])
    return np.array(final_preds)

# ===== Evaluation & Plotting =====
def evaluate_random_forest(fold_data, k_fold):
    ntrees_list = [1, 5, 10, 20, 30, 40, 50]
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for ntrees in ntrees_list:
        print(f"\n🌲 Evaluating Random Forest with {ntrees} trees")
        accs, precisions, recalls, f1s = [], [], [], []

        for i in range(k_fold):
            test_data = fold_data[fold_data["k_fold"] == i]
            train_data = fold_data[fold_data["k_fold"] != i]

            X_train = train_data.drop(columns=["label", "k_fold"])
            y_train = train_data["label"]
            X_test = test_data.drop(columns=["label", "k_fold"])
            y_test = test_data["label"]

            trees = [build_tree(*bootstrap_sample(X_train, y_train), X_train.columns) for _ in range(ntrees)]
            
            # ⭐ Save trees to JSON
            save_trees_as_json(trees, ntrees)
            predictions = random_forest_predict(trees, X_test)

            mask = predictions != None
            y_true_valid = np.array(y_test[mask], dtype=int)
            y_pred_valid = np.array(predictions[mask], dtype=int)

            accs.append(accuracy(predictions, y_test))
            precisions.append(precision(y_true_valid, y_pred_valid))
            recalls.append(recall(y_true_valid, y_pred_valid))
            f1s.append(f1_score_manual(y_true_valid, y_pred_valid))

            print(f"[Fold {i}] Acc: {accs[-1]:.4f}, Precision: {precisions[-1]:.4f}, Recall: {recalls[-1]:.4f}, F1: {f1s[-1]:.4f}")

        acc_list.append(np.mean(accs))
        prec_list.append(np.mean(precisions))
        rec_list.append(np.mean(recalls))
        f1_list.append(np.mean(f1s))

        print(f"✅ [ntrees={ntrees}] Averages → Acc: {acc_list[-1]:.4f}, Prec: {prec_list[-1]:.4f}, Rec: {rec_list[-1]:.4f}, F1: {f1_list[-1]:.4f}")

    return ntrees_list, [acc_list, prec_list, rec_list, f1_list]

def plot_metrics(ntrees_list, metrics, save_dir):
    titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    filenames = ["accuracy.png", "precision.png", "recall.png", "f1score.png"]

    for metric, title, fname in zip(metrics, titles, filenames):
        plt.figure(figsize=(6, 4))
        plt.plot(ntrees_list, metric, marker='o')
        plt.title(title)
        plt.xlabel("ntrees")
        plt.savefig(f"{save_dir}/{fname}")



if __name__ == "__main__":
    main()
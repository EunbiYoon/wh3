import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score

# ===== Decision Tree Functions =====

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

    # Best feature & its info gain
    info_gains = {f: information_gain(X, y, f) for f in features}
    best_feature = max(info_gains, key=info_gains.get)
    best_gain = info_gains[best_feature]

    # 최소 정보 이득 조건 추가
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

# ===== Cross-validation =====

def cross_validation(data, k_fold):
    class_0 = data[data['label'] == 0].reset_index(drop=True)
    class_1 = data[data['label'] == 1].reset_index(drop=True)

    fold_list = []

    for i in range(k_fold):
        class_0_start = int(len(class_0) * i / k_fold)
        class_0_end = int(len(class_0) * (i + 1) / k_fold)
        class_0_fold = class_0.iloc[class_0_start:class_0_end]

        class_1_start = int(len(class_1) * i / k_fold)
        class_1_end = int(len(class_1) * (i + 1) / k_fold)
        class_1_fold = class_1.iloc[class_1_start:class_1_end]

        fold_data = pd.concat([class_0_fold, class_1_fold]).copy()
        fold_data['k_fold'] = i
        fold_list.append(fold_data)

    return pd.concat(fold_list).reset_index(drop=True)

# ===== Helper Functions =====

def tree_to_dict(tree):
    if tree.label is not None:
        return {'label': str(tree.label)}
    return {
        'feature': str(tree.feature),
        'children': {str(value): tree_to_dict(child) for value, child in tree.children.items()}
    }

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

# ===== Main Execution =====

import matplotlib.pyplot as plt

def main():
    # 결과 저장용 리스트
    ntrees_list = [1, 5, 10, 20, 30, 40, 50]
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    data = pd.read_csv("wdbc.csv")
    data = data.rename(columns={"class": "label"})

    # numerical & categorical
    for col in data.columns:
        if col == "label":
            continue
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)  # categorical 처리
        else:
            data[col] = (data[col] > data[col].mean()).astype(int)  # numerical 처리


    k_fold = 5
    fold_data = cross_validation(data, k_fold)

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

            trees = []
            for _ in range(ntrees):
                X_boot, y_boot = bootstrap_sample(X_train, y_train)
                tree = build_tree(X_boot, y_boot, X_boot.columns)
                trees.append(tree)

            predictions = random_forest_predict(trees, X_test)

            # Filter out None predictions
            mask = predictions != None
            y_true_valid = np.array(y_test[mask], dtype=int)
            y_pred_valid = np.array(predictions[mask], dtype=int)

            acc = accuracy(predictions, y_test)
            prec = precision_score(y_true_valid, y_pred_valid, zero_division=0)
            rec = recall_score(y_true_valid, y_pred_valid, zero_division=0)
            f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0)

            accs.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            print(f"[Fold {i}] Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        # 각 ntrees에 대한 평균 성능 저장
        acc_list.append(np.mean(accs))
        prec_list.append(np.mean(precisions))
        rec_list.append(np.mean(recalls))
        f1_list.append(np.mean(f1s))

        print(f"✅ [ntrees={ntrees}] Averages → "
              f"Accuracy: {np.mean(accs):.4f}, "
              f"Precision: {np.mean(precisions):.4f}, "
              f"Recall: {np.mean(recalls):.4f}, "
              f"F1: {np.mean(f1s):.4f}")

    # 그래프 그리기
    metrics = [acc_list, prec_list, rec_list, f1_list]
    titles = ["Accuracy", "Precision", "Recall", "F1 Score"]

    plt.figure(figsize=(6, 4))
    plt.plot(ntrees_list, metrics[0], marker='o')
    plt.title("Accuracy")
    plt.xlabel("ntrees")
    plt.savefig("wdbc_result_1/accuracy.png")

    plt.figure(figsize=(6, 4))
    plt.plot(ntrees_list, metrics[1], marker='o')
    plt.title("Precision")
    plt.xlabel("ntrees")
    plt.savefig("wdbc_result_1/precision.png")

    plt.figure(figsize=(6, 4))
    plt.plot(ntrees_list, metrics[2], marker='o')
    plt.title("Recall")
    plt.xlabel("ntrees")
    plt.savefig("wdbc_result_1/recall.png")

    plt.figure(figsize=(6, 4))
    plt.plot(ntrees_list, metrics[3], marker='o')
    plt.title("F1 Score")
    plt.xlabel("ntrees")
    plt.savefig("wdbc_result_1/f1score.png")



if __name__ == "__main__":
    main()

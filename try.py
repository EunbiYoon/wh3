import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import math
import random

# ===== Main Function =====
def main():
    # Load dataset and preprocess it
    data = load_and_preprocess_data("raisin.csv")
    # Apply stratified k-fold cross-validation
    fold_data = cross_validation(data, k_fold=5)

    # Train and evaluate Random Forest
    ntrees_list, metrics = evaluate_random_forest(fold_data, k_fold=5)
    # Plot evaluation metrics
    plot_metrics(ntrees_list, metrics, save_dir="raisin_result")

# ===== Preprocessing =====
def load_and_preprocess_data(filepath):
    # Load CSV and rename target column
    data = pd.read_csv(filepath)
    data = data.rename(columns={"class": "label"})

    # Process each column by its suffix
    for col in data.columns:
        if col == "label":
            continue
        elif col.endswith("_cat"):
            data[col] = data[col].astype(str)  # Categorical feature
        elif col.endswith("_num"):
            data[col] = pd.to_numeric(data[col])  # Numerical feature
        else:
            print(f"There is an error in csv file column name: {col}")
    return data

# ===== Cross-validation =====
def cross_validation(data, k_fold):
    # Separate data by class label for stratified sampling
    class_0 = data[data['label'] == 0].reset_index(drop=True)
    class_1 = data[data['label'] == 1].reset_index(drop=True)

    all_data = pd.DataFrame()
    for i in range(k_fold):
        # Slice each class proportionally into folds
        class_0_start = int(len(class_0) * i / k_fold)
        class_0_end = int(len(class_0) * (i + 1) / k_fold)
        class_1_start = int(len(class_1) * i / k_fold)
        class_1_end = int(len(class_1) * (i + 1) / k_fold)

        class_0_fold = class_0.iloc[class_0_start:class_0_end]
        class_1_fold = class_1.iloc[class_1_start:class_1_end]

        fold_data = pd.concat([class_0_fold, class_1_fold]).copy()
        fold_data["k_fold"] = i

        all_data = pd.concat([all_data, fold_data], ignore_index=True)
    return all_data

# ===== Evaluation & Plotting =====
def evaluate_random_forest(fold_data, k_fold):
    ntrees_list = [1, 5, 10, 20, 30, 40, 50]
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    # operate in all ntrees
    for ntrees in ntrees_list:
        print(f"\nEvaluating Random Forest with {ntrees} trees")
        accs, precisions, recalls, f1s = [], [], [], []

        # execute cross_validation
        for i in range(k_fold):
            # Split fold into training and testing
            test_data = fold_data[fold_data["k_fold"] == i]
            train_data = fold_data[fold_data["k_fold"] != i]

            # Separate features(attribute) and labels -> boostrap
            X_train = train_data.drop(columns=["label", "k_fold"])
            y_train = train_data["label"]
            X_test = test_data.drop(columns=["label", "k_fold"])
            y_test = test_data["label"]

            # Train Random Forest
            trees = []
            for _ in range(ntrees):
                # sampling boostrap data
                X_sample, y_sample = bootstrap_sample(X_train, y_train)
                tree = build_tree(X_sample, y_sample, X_train.columns)
                trees.append(tree)

            # Save trees and make predictions
            save_trees_as_json(trees, ntrees)
            predictions = random_forest_predict(trees, X_test)

            # Filter valid predictions
            mask = predictions != None
            y_true_valid = np.array(y_test[mask], dtype=int)
            y_pred_valid = np.array(predictions[mask], dtype=int)

            # Evaluate metrics
            accs.append(accuracy(predictions, y_test))
            precisions.append(precision(y_true_valid, y_pred_valid))
            recalls.append(recall(y_true_valid, y_pred_valid))
            f1s.append(f1_score_manual(y_true_valid, y_pred_valid))

            print(f"[Fold {i}] Acc: {accs[-1]:.4f}, Precision: {precisions[-1]:.4f}, Recall: {recalls[-1]:.4f}, F1: {f1s[-1]:.4f}")

        acc_list.append(np.mean(accs))
        prec_list.append(np.mean(precisions))
        rec_list.append(np.mean(recalls))
        f1_list.append(np.mean(f1s))

        print(f"Average Results for ntrees={ntrees} => Acc: {acc_list[-1]:.4f}, Prec: {prec_list[-1]:.4f}, Rec: {rec_list[-1]:.4f}, F1: {f1_list[-1]:.4f}")

    return ntrees_list, [acc_list, prec_list, rec_list, f1_list]

# Draw bootstrap sample from training data 
def bootstrap_sample(X, y):
    # random select on index
    idxs = np.random.choice(len(X), size=len(X), replace=True) # replacement = True
    # get the selected index in X,y
    X_sample = X.iloc[idxs].reset_index(drop=True)
    y_sample = y.iloc[idxs].reset_index(drop=True)
    # return same as input
    return X_sample, y_sample


def build_tree(X, y, features):
    # Setting default value
    depth=0
    max_depth=5
    min_info_gain=1e-5

    # Stopping criteria for decision tree
    if len(y.unique()) == 1 or len(features) == 0 or depth == max_depth:
        return Node(label=y.mode()[0])

    # Select m random attributes (m ≈ sqrt(#features))
    m = max(1, int(math.sqrt(len(features))))
    selected_features = random.sample(list(features), m)

    best_feature, best_gain, best_threshold = None, -1, None

    for f in selected_features:
        if X[f].dtype == 'object':
            # Evaluate gain for categorical feature
            values = X[f].unique()
            subsets = [y[X[f] == v] for v in values]
            gain = entropy(y) - sum((len(sub)/len(y)) * entropy(sub) for sub in subsets)
            if gain > best_gain:
                best_feature, best_gain, best_threshold = f, gain, None
        else:
            # Evaluate gain for numerical feature (using mean as threshold)
            threshold = X[f].mean()
            left_y = y[X[f] <= threshold]
            right_y = y[X[f] > threshold]
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            # decision tree split by information gain
            gain = information_gain_split(left_y, right_y)
            if gain > best_gain:
                best_feature, best_gain, best_threshold = f, gain, threshold

    if best_gain < min_info_gain or best_feature is None:
        return Node(label=y.mode()[0])

    tree = Node(feature=best_feature, threshold=best_threshold)

    if best_threshold is None:
        # Categorical split
        for value in X[best_feature].unique():
            subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
            subset_y = y[X[best_feature] == value]
            tree.children[value] = build_tree(subset_X, subset_y, [f for f in features if f != best_feature], depth + 1, max_depth, min_info_gain)
    else:
        # Numerical split
        left_mask = X[best_feature] <= best_threshold
        right_mask = X[best_feature] > best_threshold
        tree.children["<="] = build_tree(X[left_mask], y[left_mask], features, depth + 1, max_depth, min_info_gain)
        tree.children[">"] = build_tree(X[right_mask], y[right_mask], features, depth + 1, max_depth, min_info_gain)

    return tree

def tree_to_dict(node):
    # Serialize tree to dictionary format
    if node.label is not None:
        return {"label": int(node.label)}
    return {
        "feature": node.feature,
        "threshold": node.threshold,
        "children": {str(k): tree_to_dict(v) for k, v in node.children.items()}
    }


def save_trees_as_json(trees, ntrees, base_dir="saved_trees"):
    # Save all trees in the forest as JSON files
    folder = os.path.join(base_dir, f"ntrees_{ntrees}")
    os.makedirs(folder, exist_ok=True)
    for i, tree in enumerate(trees, start=1):
        tree_dict = tree_to_dict(tree)
        with open(os.path.join(folder, f"tree_{i}.json"), "w") as f:
            json.dump(tree_dict, f, indent=4)

def random_forest_predict(trees, X):
    # Predict by majority voting from all trees in the forest
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

def predict(tree, X):
    # Predict labels for each row in dataset using a single decision tree
    predictions = []
    for _, row in X.iterrows():
        node = tree
        while node.label is None:
            val = row[node.feature]
            if node.threshold is not None:
                if val <= node.threshold:
                    node = node.children.get("<=")
                else:
                    node = node.children.get(">")
            else:
                if val not in node.children:
                    node = None
                    break
                node = node.children[val]
        predictions.append(node.label if node else None)
    return np.array(predictions)

def accuracy(predictions, true_labels):
    # Calculate accuracy
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    valid = predictions != None
    return np.mean(predictions[valid] == true_labels[valid])

def precision(y_true, y_pred):
    # Calculate precision
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    # Calculate recall
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score_manual(y_true, y_pred):
    # Calculate F1-score
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def plot_metrics(ntrees_list, metrics, save_dir):
    # Plot accuracy, precision, recall, and F1 score vs. number of trees
    titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    filenames = ["accuracy.png", "precision.png", "recall.png", "f1score.png"]
    os.makedirs(save_dir, exist_ok=True)
    for metric, title, fname in zip(metrics, titles, filenames):
        plt.figure(figsize=(6, 4))
        plt.plot(ntrees_list, metric, marker='o')
        plt.title(title)
        plt.xlabel("ntrees")
        plt.ylabel(title)
        plt.grid(True)
        plt.savefig(f"{save_dir}/{fname}")
        plt.close()

# ===== Entropy & Tree Node Class =====
def entropy(y):
    # Compute entropy of label distribution
    class_counts = y.value_counts()
    probabilities = class_counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain_split(y_left, y_right):
    # Compute information gain from a binary split
    total_entropy = entropy(pd.concat([y_left, y_right]))
    weighted_entropy = (len(y_left)/ (len(y_left) + len(y_right))) * entropy(y_left) + \
                       (len(y_right)/ (len(y_left) + len(y_right))) * entropy(y_right)
    return total_entropy - weighted_entropy

class Node:
    # Node in the decision tree
    def __init__(self, feature=None, threshold=None, label=None, children=None):
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.children = children if children else {}

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle

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

def build_tree(X, y, features):
    if len(y.unique()) == 1:
        return Node(label=y.iloc[0])
    if len(features) == 0:
        return Node(label=y.mode()[0])

    best_feature = max(features, key=lambda f: information_gain(X, y, f))
    tree = Node(feature=best_feature)
    remaining_features = [f for f in features if f != best_feature]

    for value in X[best_feature].unique():
        subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
        subset_y = y[X[best_feature] == value]
        tree.children[value] = build_tree(subset_X, subset_y, remaining_features)

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
    return np.mean(predictions == true_labels)

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

# ===== Tree to JSON Serializable Dict =====

def tree_to_dict(tree):
    if tree.label is not None:
        return {'label': str(tree.label)}  # Ensures label is JSON serializable
    return {
        'feature': str(tree.feature),  # Ensures feature is string
        'children': {str(value): tree_to_dict(child) for value, child in tree.children.items()}
    }

# ===== Main Execution =====

def main():
    data = pd.read_csv("wdbc.csv")
    data = data.rename(columns={"class": "label"})

    # Optional: Convert features to binary (0/1) based on mean — improves splits
    for col in data.columns:
        if col != "label":
            data[col] = (data[col] > data[col].mean()).astype(int)

    k_fold = 5
    fold_data = cross_validation(data, k_fold)

    accuracies = []

    for i in range(k_fold):
        test_data = fold_data[fold_data["k_fold"] == i]
        train_data = fold_data[fold_data["k_fold"] != i]
        print(f"test_data: {len(test_data)}")

        X_train = train_data.drop(columns=["label", "k_fold"])
        y_train = train_data["label"]
        X_test = test_data.drop(columns=["label", "k_fold"])
        y_test = test_data["label"]

        tree = build_tree(X_train, y_train, X_train.columns)
        predictions = predict(tree, X_test)

        acc = accuracy(predictions, y_test)
        accuracies.append(acc)

        print(f"[Fold {i}] Accuracy: {acc:.4f}")

        if i == 0:
            with open("decision_tree.json", 'w') as f:
                json.dump(tree_to_dict(tree), f, indent=4)

    print(f"\n✅ Average Accuracy over {k_fold} folds: {np.mean(accuracies):.4f}")

if __name__ == "__main__":
    main()

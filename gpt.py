# random_forest_hw3.py

import numpy as np
import pandas as pd
import math
import random
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# --- Helper: Entropy & Information Gain ---
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log10(p) for p in probabilities if p > 0])

def information_gain(X_column, y, threshold):
    left_mask = X_column <= threshold
    right_mask = ~left_mask
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0
    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])
    total = len(y)
    weighted_avg = (np.sum(left_mask) * left_entropy + np.sum(right_mask) * right_entropy) / total
    return entropy(y) - weighted_avg

# --- Decision Tree Node ---
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-7, num_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.num_features = num_features

    def fit(self, X, y, depth=0):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y, depth)

    def _best_split(self, X, y, features):
        best_gain = -1
        split_idx, split_threshold = None, None

        for idx in features:
            thresholds = np.unique(X[:, idx])
            for t in thresholds:
                gain = information_gain(X[:, idx], y, t)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_threshold = t

        return split_idx, split_threshold, best_gain

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_labels = len(set(y))

        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            most_common = Counter(y).most_common(1)[0][0]
            return most_common

        features = random.sample(range(num_features), self.num_features or int(math.sqrt(num_features)))
        best_idx, threshold, best_gain = self._best_split(X, y, features)

        if best_gain < self.min_gain:
            return Counter(y).most_common(1)[0][0]

        left_mask = X[:, best_idx] <= threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return (best_idx, threshold, left_subtree, right_subtree)

    def predict_one(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            idx, threshold, left, right = node
            if inputs[idx] <= threshold:
                node = left
            else:
                node = right
        return node

    def predict(self, X):
        return np.array([self.predict_one(row) for row in X])

# --- Random Forest ---
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, min_gain=1e-7):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gain=self.min_gain,
                num_features=int(math.sqrt(X.shape[1]))
            )
            X_samp, y_samp = self.bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)

# --- Evaluation Function ---
def evaluate_random_forest(X, y, n_trees_list, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    metrics = {"ntree": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    for ntree in n_trees_list:
        accs, precs, recs, f1s = [], [], [], []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            rf = RandomForest(n_trees=ntree)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred))
            recs.append(recall_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))

        metrics["ntree"].append(ntree)
        metrics["accuracy"].append(np.mean(accs))
        metrics["precision"].append(np.mean(precs))
        metrics["recall"].append(np.mean(recs))
        metrics["f1"].append(np.mean(f1s))

    return metrics

# --- Plotting Function ---
def plot_metrics(metrics):
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plt.figure()
        plt.plot(metrics["ntree"], metrics[metric], marker='o')
        plt.xlabel("Number of Trees (ntree)")
        plt.ylabel(metric.capitalize())
        plt.title(f"Random Forest - {metric.capitalize()} vs ntree")
        plt.grid(True)
        plt.savefig(f"{metric}_vs_ntree.png")

# --- Example Usage with Preprocessed Data ---
# Replace this part with actual data loading in your workflow
if __name__ == "__main__":
    # Dummy example with sklearn breast cancer dataset
    data = pd.read_csv('wdbc.csv')
    y = data['label']  # 또는 정확한 열 이름
    X = data.drop(columns=['label'])

    n_trees_list = [1, 5, 10, 20, 30, 40, 50]
    results = evaluate_random_forest(X, y, n_trees_list)
    plot_metrics(results)

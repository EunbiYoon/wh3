import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import math
import random

# ===== Main Function =====
def main():
    ##### choose dataset #####
    filename = "datasets/raisin.csv" # <<<<===== changing point 
    base = os.path.basename(filename)     
    basename = os.path.splitext(base)[0]

    # Load dataset and preprocess it
    data = load_and_preprocess_data(filename)
    
    # Apply stratified k-fold cross-validation
    fold_data = cross_validation(data, k_fold=5)

    # Train and evaluate Random Forest
    ntrees_list, metrics = evaluate_random_forest(fold_data, k_fold=5)
    
    # Plot evaluation metrics
    plot_metrics(basename, ntrees_list, metrics, save_dir="plots")

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

    # proceed seperate class_0(label=0 sub dataset), class_1(label=1 sub dataset) to preserve proportions
    all_data = pd.DataFrame()
    for i in range(k_fold):
        # Slice each class proportionally into folds => satisfied disjoint condition 
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


##################################### make decision tree ##################################### 
# ===== Entropy & Tree Node Class =====
class Node:
    # Node in the decision tree
    def __init__(self, feature=None, threshold=None, label=None, children=None):
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.children = children if children else {}

# Compute entropy of label distribution
# e.g., (x=sunny)-> y = [y,y,y,n,n] -> entropy calcuate
# e.g., all y = [y,y,y,n,n,n,n,n,y,y,y,n] -> entropy calcuate
def entropy(y):
    # count label and change to series
    class_counts = y.value_counts()
    # divided by total length
    probabilities = class_counts / len(y)
    # entropy = sum(- prob * log2(prob))
    return -np.sum(probabilities * np.log2(probabilities))


def build_tree(X, y, features, depth=0):
    ####### Stop splitting node criteria (maximal_depth and minimal_gain are combined) ####### 
    # depth = 0 --> check current node depth to confirm when depth reached to max_depth
    max_depth=5 # maximal_depth
    min_info_gain=1e-5 # minimal_gain

    ### Check Stop Spliting condition ===> maximal_depth
    # unique -> check all the same class
    # len(features) -> no features left
    # current depth = max_depth
    if len(y.unique()) == 1 or len(features) == 0 or depth == max_depth:
        return Node(label=y.mode()[0])

    ### Select m random attributes
    m = int(math.sqrt(len(features))) # m ≈ sqrt(#features)
    selected_features = random.sample(list(features), m) # select the mth sample in features
    # branch splitting criteria, best_gain=-1 -> if gain > best_gain(-1) => always true for first 
    best_feature, best_gain, best_threshold = None, -1, None 


    ####### select features based on information gain ####### 
    for feature in selected_features:
        # categorical attribute -> best_threshold : no need
        if feature.endswith('_cat'):
            # Evaluate gain for categorical feature
            values = X[feature].unique() # unique values in attribute

            ### information gain 구하기
            subsets = []
            weighted_entropy = 0
            # try all values in feature
            for v in values:
                # select X(feature)=v and get the y(label) => subset_y = y[X['gender_cat'] == 'female']
                subset_y = y[X[feature] == v]
                # subsets = the corresponding y values for each unique value of the feature
                subsets.append(subset_y)
                # one value in featrues y values / all y values 
                proportion = len(subset_y) / len(y)
                # add all entropy from each values 
                weighted_entropy = weighted_entropy + proportion * entropy(subset_y)
            gain = entropy(y) - weighted_entropy

            ###### get the best features ######
            # threshold doesnt need because this is categorical attribute
            # default best_gain=-1 -> 1st feature selected but from 2nd feature, need to compare with best feature(1st feature)
            if gain > best_gain:
                best_feature, best_gain, best_threshold = feature, gain, None

        # numerical attribute -> best_threshold : need
        else:
            # threshold is selected as average value in features.
            threshold = X[feature].mean()
            # left : same or lower than average / right : higher than average
            # X[feature] <= threshold : [True, True, False, True, True, False]
            # y[X[features] <= threshold] : y[[True, True, False, True, True, False]] = y[(index)0,1,3,4] = [(y_value)1,1,1,0]
            left_y = y[X[feature] <= threshold]
            right_y = y[X[feature] > threshold]
            
            # after splitting, if one side is empty then skip and find another attribute
            if len(left_y) == 0 or len(right_y) == 0:          
                # continue : skip this loop and go to next iteration
                continue
            
            # proportions
            total_len = len(y)
            left_prop = len(left_y) / total_len
            right_prop = len(right_y) / total_len

            # weighted entropy
            weighted_entropy = left_prop * entropy(left_y) + right_prop * entropy(right_y)
            gain = entropy(y) - weighted_entropy

            ###### get the best features ######
            # update best split if gain improves
            if gain > best_gain:
                best_feature, best_gain, best_threshold = feature, gain, threshold


    ####### before saved into Node to check stopping criteria ====> minial_gain
    if best_gain < min_info_gain or best_feature is None: 
        # make leaf node and no more branching
        return Node(label=y.mode()[0])

    # after for loop, save feature, threshold to Node
    # threshold become standard value for splitting
    tree = Node(feature=best_feature, threshold=best_threshold)


    ####### build tree based on best_features, best_threshold ####### 
    # Categorical attribute(features)
    if best_threshold is None:
        for value in X[best_feature].unique():
            # get data only which column=best_feature is "value" and drop the best_feature column
            subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
            # get data only which column=best_feature is "value"
            subset_y = y[X[best_feature] == value]
            ########## make subtree and connect to child node ##########
            # new features
            new_features = []
            # f in features
            for f in features:
                if f != best_feature:
                    new_features.append(f)
            # excluding best_feature, create new_featrues list and operate build_tree function again
            child_subtree = build_tree(subset_X,subset_y,new_features,depth + 1)
            # connect to child node
            tree.children[value] = child_subtree
    
    # Numerical attribute
    else:
        # mask : depends on each instance achieve condition
        # mask = [False, True, False, False, False, True]
        left_mask = X[best_feature] <= best_threshold
        right_mask = X[best_feature] > best_threshold
        
        # make left and right branch
        # use boolean indexing, select index(instance) which is true
        tree.children["<="] = build_tree(X[left_mask], y[left_mask], features, depth + 1)
        tree.children[">"] = build_tree(X[right_mask], y[right_mask], features, depth + 1)
    
    # final tree is returned by checking stopping criteria
    return tree


##################################### make prediction ##################################### 
# Predict by majority voting from all trees in the forest
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

 # Predict labels for each row in dataset using a single decision tree
def predict(tree, X):
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

##################################### accuracy, precision, recall, f1_score ##################################### 
# Calculate accuracy
def accuracy(predictions, true_labels):
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    valid = predictions != None
    return np.mean(predictions[valid] == true_labels[valid])

# Calculate precision
def precision(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

# Calculate recall
def recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# Calculate F1-score
def f1_score_manual(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

##################################### plot the metrics ##################################### 
def plot_metrics(dataset_name, ntrees_list, metrics, save_dir):
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
        plt.savefig(f"{save_dir}/{dataset_name}_{fname}")
        plt.close()

##################################### tree -> json file ##################################### 
# Serialize tree to dictionary format
def tree_to_dict(node):
    if node.label is not None:
        return {"label": int(node.label)}
    return {
        "feature": node.feature,
        "threshold": node.threshold,
        "children": {str(k): tree_to_dict(v) for k, v in node.children.items()}
    }

# Save all trees in the forest as JSON files
def save_trees_as_json(trees, ntrees, base_dir="saved_trees"):
    folder = os.path.join(base_dir, f"ntrees_{ntrees}")
    os.makedirs(folder, exist_ok=True)
    for i, tree in enumerate(trees, start=1):
        tree_dict = tree_to_dict(tree)
        with open(os.path.join(folder, f"tree_{i}.json"), "w") as f:
            json.dump(tree_dict, f, indent=4)

if __name__ == "__main__":
    main()

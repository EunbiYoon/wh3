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
    filename = "datasets/loan.csv" # <<<<===== changing point 
    base = os.path.basename(filename)     
    basename = os.path.splitext(base)[0]

    # Load dataset and preprocess it
    data = load_and_preprocess_data(filename)
    
    # Apply stratified k-fold cross-validation
    fold_data = cross_validation(data, k_fold=5)

    # Train and evaluate Random Forest
    ntrees_list, metrics = evaluate_random_forest(fold_data, 5, basename)
    
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
    class_0 = data[data['label'] == 0].sample(frac=1).reset_index(drop=True) # sampling with fraction=100%
    class_1 = data[data['label'] == 1].sample(frac=1).reset_index(drop=True) # sampling with fraction=100%

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


# Draw bootstrap sample from training data 
def bootstrap_sample(X, y):
    # random select on index
    idxs = np.random.choice(len(X), size=len(X), replace=True) # replacement = True
    # get the selected index in X,y
    X_sample = X.iloc[idxs].reset_index(drop=True)
    y_sample = y.iloc[idxs].reset_index(drop=True)
    # return same as input
    return X_sample, y_sample


# ===== Evaluation & Plotting =====
def evaluate_random_forest(fold_data, k_fold, basename):
    ntrees_list = [1, 5, 10, 20, 30, 40, 50]
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    # operate in all ntrees
    for ntrees in ntrees_list:
        print(f"\nEvaluating Random Forest with {ntrees} trees")
        accs, precisions, recalls, f1s = [], [], [], []

        # execute cross_validation
        for i in range(k_fold):
            # Split fold into training and testing
            test_data = fold_data[fold_data["k_fold"] == i] # test = select k_fold data
            train_data = fold_data[fold_data["k_fold"] != i] # train = select all data but not k_fold data

            # Separate features(attribute) and labels for boostrap
            X_train = train_data.drop(columns=["label", "k_fold"]) # get only attribute
            y_train = train_data["label"] # get only label
            X_test = test_data.drop(columns=["label", "k_fold"]) # get only attribute
            y_test = test_data["label"] # get only label
        
            # Train Random Forest
            trees = []
            # repeat by tree count(ntrees)
            for index in range(ntrees): 
                # sampling bootstrap for train data
                X_sample, y_sample = bootstrap_sample(X_train, y_train) ###### only train data, boostrap datasets by sampling (with replacement)
                # make tree with bootstrap data
                tree = build_tree(X_sample, y_sample, X_train.columns) #### each boostrap makes each tree
                # add tree in list of trees
                trees.append(tree)

            # Save trees and make predictions
            predictions = random_forest_predict(trees, X_test)

            # Save tree as json format to check later (tree built based on training data)
            save_trees_as_json(trees, ntrees)

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

        # make result as list
        acc_list.append(np.mean(accs))
        prec_list.append(np.mean(precisions))
        rec_list.append(np.mean(recalls))
        f1_list.append(np.mean(f1s))
        print(f"Average Results for ntrees={ntrees} => Acc: {acc_list[-1]:.4f}, Prec: {prec_list[-1]:.4f}, Rec: {rec_list[-1]:.4f}, F1: {f1_list[-1]:.4f}")

        # make result as dataframe
        # make result as dataframe
        result = pd.DataFrame({
            f"ntrees={nt}": [acc, prec, rec, f1]
            for nt, acc, prec, rec, f1 in zip(ntrees_list, acc_list, prec_list, rec_list, f1_list)
        }, index=["Accuracy", "Precision", "Recall", "F1Score"])

        # Add 'Average' column
        result["Average"] = result.mean(axis=1)

    # Save to Excel
    result.to_excel(f"result/{basename}.xlsx")
    return ntrees_list, [acc_list, prec_list, rec_list, f1_list]



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
    selected_features = random.sample(list(features), m) # select the mth sample in features with m counts
    # branch splitting criteria, best_gain=-1 -> if gain > best_gain(-1) => always true for first 
    best_feature, best_gain, best_threshold = None, -1, None # initial setting before for loop

    ####### select features based on information gain ####### 
    for feature in selected_features:
        # categorical attribute -> best_threshold : no need
        if feature.endswith('_cat'):
            ### information gain 구하기
            # Evaluate gain for categorical feature
            values = X[feature].unique() # unique values in attribute
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
            ### information gain 구하기
            # threshold is selected as average value in features.
            threshold = X[feature].mean() ## threshold ==> average use! 
            # left : same or lower than average / right : higher than average
            # X[feature] <= threshold : [True, True, False, True, True, False]
            # y[X[features] <= threshold] : y[[True, True, False, True, True, False]] = y[(index)0,1,3,4] = [(y_value)1,1,1,0]
            left_y = y[X[feature] <= threshold]
            right_y = y[X[feature] > threshold]
            
            # after splitting, if one side is empty then skip and find another attribute
            if len(left_y) == 0 or len(right_y) == 0:          
                return Node(label=y.mode()[0])  # stop split and make current node leaf
            
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
            # new features ==> usually categorical attribute, not using again
            new_features = []
            # f in features
            for f in features:
                if f != best_feature:
                    new_features.append(f)
            # excluding best_feature, create new_featrues list and operate build_tree function again
            child_subtree = build_tree(subset_X,subset_y,new_features,depth + 1) ## ===> next iteration
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
        # no need new features ==> usually numerical attribute, using again because threshold can be changed
        tree.children["<="] = build_tree(X[left_mask], y[left_mask], features, depth + 1)  ## ===> next iteration
        tree.children[">"] = build_tree(X[right_mask], y[right_mask], features, depth + 1)  ## ===> next iteration
    
    # final tree is returned by checking stopping criteria
    return tree


##################################### make prediction ##################################### 
# Predict by majority voting from all trees in the forest
def random_forest_predict(trees, X_test):
    # Collect predictions from all trees (shape: [num_trees, num_X_test_samples])
    # k-fold function result => test dataset fixed
    # test sample : 3, tree: 4 => [[1,0,0],[1,1,1],[0,1,1],[1,1,1]]
    tree_preds = np.array([predict(tree, X_test) for tree in trees])
    # Total number of test samples
    test_all = X_test.shape[0]  
    # List to store final predictions after majority voting
    final_preds = [] 

    # Iterate over each sample
    for i in range(test_all):
        # tree_preds[:, i] ==> ith sample collection
        # test sample : 3, tree: 4 => [[1,0,0],[1,1,1],[0,1,1],[1,1,1]] ====> tree_preds[:, 0]=[1,1,0,1] ==> extract 0th values in list
        row_preds = tree_preds[:, i]  # Get predictions for the i-th sample from all trees

        ### Majority Voting
        # Filter out None values and count occurrences of each predicted class
        # Result :: values=[0, 1] counts =[1, 3]
        values, counts = np.unique(row_preds[row_preds != None], return_counts=True)

        # row_preds = [None, None] ---> Result :: value=[] count=[] -> len(count)=0
        if len(counts) == 0:
            # If no valid predictions, append None
            final_preds.append(None)
        elif len(counts) > 1 and counts[0] == counts[1]:
            final_preds.append(random.choice(values))
        else:
            # Select the class with the highest vote (majority voting) -> argmax
            # argmax : return biggest value's index
            final_preds.append(values[np.argmax(counts)])

    # Return the final predictions as a numpy array
    return np.array(final_preds)


# Predict labels for each row in dataset using a single decision tree
def predict(tree, X_test):
    # gather all the prediction results
    predictions = []

    ##### repeat X_test counts #####
    # get index(not using), row(samples) from itterows
    for index, row in X_test.iterrows(): # iterate row by row
        # initialize the node = tree
        # each sample starts to search from root of tree
        node = tree
        # repeat until reaching out to the leaf node (label exist -> leaf node)
        # input value into tree's node
        while node.label is None:
            # get the current node feature's value
            val = row[node.feature]

            # if "categorical" attribute
            if node.threshold is None:
                if val in node.children: # val in children node -> follow children node => become current node
                    node = node.children[val]
                else:
                    node = None # no branch to follow
                    break

            # if "numerical" attribute
            else:
                if val <= node.threshold:
                    # move to left child node
                    node = node.children.get("<=")
                else:
                    # move to right child node
                    node = node.children.get(">")

        ##### X_test counts -> each test sample prediction #####
        if node is not None: # final prediction
            predictions.append(node.label)
        else:
            predictions.append(None)
            
    # return prediction as array
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
# def plot_metrics(basename, ntrees_list, metrics, save_dir):
#     titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
#     filenames = ["accuracy.png", "precision.png", "recall.png", "f1score.png"]
#     os.makedirs(save_dir, exist_ok=True)
#     for metric, title, fname in zip(metrics, titles, filenames):
#         plt.figure(figsize=(6, 4))
#         plt.plot(ntrees_list, metric, marker='o')
#         plt.title(f"Dataset : {basename.capitalize()} / Metrics : "+title)
#         plt.xlabel("ntrees")
#         plt.ylabel(title)
#         plt.grid(True)
#         plt.savefig(f"{save_dir}/{basename}_{fname}")
#         plt.close()


def plot_metrics(basename, ntrees_list, metrics, save_dir):
    titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))  # 전체 크기 설정 (2행 2열 서브플롯)
    
    for i, (metric, title) in enumerate(zip(metrics, titles), 1):
        plt.subplot(2, 2, i)  # 2행 2열 중 i번째 subplot
        plt.plot(ntrees_list, metric, marker='o')
        plt.title(title)
        plt.xlabel("ntrees")
        plt.grid(True)
    
    # 제목 대문자화 및 폰트 크기 17로 설정
    suptitle_text = f"Dataset: {basename.capitalize()}"
    plt.suptitle(suptitle_text, fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 제목과 서브플롯 간 간격 조정
    plt.savefig(f"{save_dir}/{basename}.png")
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

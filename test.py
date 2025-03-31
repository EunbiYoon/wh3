import pandas as pd
import numpy as np
import openpyxl


def cross_validation(data, k_fold):
    # split by two class
    class_0 = data[data['label'] == 0].reset_index(drop=True)
    class_1 = data[data['label'] == 1].reset_index(drop=True)
    print(33)
    print(f"class_0: {class_0} / class_1: {class_1}")

    fold_list = []

    for i in range(k_fold):
        class_0_start = int(len(class_0) * i / k_fold)
        class_0_end = int(len(class_0) * (i + 1) / k_fold)
        class_0_fold = class_0.iloc[class_0_start:class_0_end]

        class_1_start = int(len(class_1) * i / k_fold)
        class_1_end = int(len(class_1) * (i + 1) / k_fold)
        class_1_fold = class_1.iloc[class_1_start:class_1_end]

        fold_data = pd.concat([class_0_fold, class_1_fold]).copy()
        fold_data['k_fold'] = i  # 현재 fold 번호 추가

        fold_list.append(fold_data)

    fold_total = pd.concat(fold_list).reset_index(drop=True)
    fold_total.to_excel('33.xlsx')
    print(fold_total)
    return fold_total


def main():
    data = pd.read_csv('raisin.csv')
    print(data.head())
    cross_validation(data, 5)


if __name__ == "__main__":
    main()
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from find_best_parameters import find_best_parameters

def plot_balance_class(classes):
    unique, counts = np.unique(classes, return_counts=True)
    plt.bar(unique, counts)
    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()


def load_data(dataset_path: str):
    """Load data and convert class type to str
    """
    data = arff.loadarff(dataset_path)
    df = pd.DataFrame(data[0])
    df['class'] = df['class'].astype(str)
    return df

def balance_class(df):
    max_size = df['class'].value_counts().max()
    lst = [df]
    for class_index, group in df.groupby('class'):
        lst.append(group.sample(max_size-len(group), replace=True))
    frame_new = pd.concat(lst)
    return frame_new

def main():
    """Main function
    """
    df = load_data('diabetes.arff')
    features_cols = [i for i in df.columns.values.tolist() if i not in [
        'class']]
    x_train, x_test, y_train, y_test = train_test_split(
        df[features_cols], df['class'], test_size=0.20, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred) * 100
    print(acc_score)
    """ x_train['class'] = y_train
    x_train = balance_class(x_train)
    y_train = x_train['class']
    x_train.pop('class') """
    max_depth, _ = find_best_parameters('max_depth', list(range(1, 30)), x_train, x_test, y_train, y_test)
    print("Best max_depth t: ", max_depth)
    min_samples_split, _ = find_best_parameters('min_samples_split', list(range(2, 400, 5)),
        x_train, x_test, y_train, y_test)
    min_samples_split = int(min_samples_split)
    print("Best min samples split: ", min_samples_split)
    min_samples_leaf, _ = find_best_parameters('min_samples_leaf',  list(range(2, 200, 5)),
        x_train, x_test, y_train, y_test)
    min_samples_leaf = int(min_samples_leaf)
    print("Best sample leaf: ", min_samples_leaf)
    max_leaf_nodes, _ = find_best_parameters('max_leaf_nodes', list(range(2, 100)),
        x_train, x_test, y_train, y_test)
    max_leaf_nodes = int(max_leaf_nodes)
    print("Best max leaf nodes split: ", max_leaf_nodes)
    min_impurity_decrease, _ = find_best_parameters('min_impurity_decrease', [
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.15,
        0.2,
        0.4],
        x_train, x_test, y_train, y_test)
    print("Best min impurity decrease: ", min_impurity_decrease)
    clf = DecisionTreeClassifier(min_impurity_decrease=min_impurity_decrease, max_depth=max_depth, 
    min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, random_state=42)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred) * 100
    print(acc_score)


if __name__ == "__main__":
    main()

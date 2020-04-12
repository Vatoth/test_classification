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
from tqdm import tqdm_notebook as tqdm


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

def max_depth_parameters(x_train, x_test, y_train, y_test):
    c_parameter_name = 'max_depth_sparameters'
    c_parameter_values = list(range(1, 30))
    df = pd.DataFrame(columns=[c_parameter_name, 'accuracy'])
    for input_parameter in c_parameter_values:
        clf = DecisionTreeClassifier(
            max_depth=input_parameter, random_state=42)
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        df = df.append(
            {c_parameter_name: input_parameter, 'accuracy': acc_score},
            ignore_index=True)
    plt.subplot(111)
    sns.pointplot(x=c_parameter_name, y="accuracy", data=df)
    title = 'Model Accuracy(%) vs ' + c_parameter_name + ' parameter'
    plt.title(title)
    plt.xticks(rotation=90)
    plt.grid()

    return df.loc[df['accuracy'].idxmax()][c_parameter_name]


def min_samples_split_parameters(x_train, x_test, y_train, y_test):
    c_parameter_name = 'min_samples_split'
    c_parameter_values = list(range(5, 400, 5))
    df = pd.DataFrame(columns=[c_parameter_name, 'accuracy'])
    for input_parameter in c_parameter_values:
        clf = DecisionTreeClassifier(
            min_samples_split=input_parameter,
            random_state=42)
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        df = df.append(
            {c_parameter_name: input_parameter, 'accuracy': acc_score},
            ignore_index=True)
    plt.subplot(211)
    sns.pointplot(x=c_parameter_name, y="accuracy", data=df)

    title = 'Model Accuracy(%) vs ' + c_parameter_name + ' parameter'
    plt.title(title)
    plt.xticks(rotation=90)
    plt.grid()

    return df.loc[df['accuracy'].idxmax()][c_parameter_name]


def min_samples_leaf_parameters(x_train, x_test, y_train, y_test):
    c_parameter_name = 'min_samples_leaf'
    c_parameter_values = list(range(5, 200, 5))
    df = pd.DataFrame(columns=[c_parameter_name, 'accuracy'])
    for input_parameter in c_parameter_values:
        clf = DecisionTreeClassifier(
            min_samples_leaf=input_parameter,
            random_state=42)
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        df = df.append(
            {c_parameter_name: input_parameter, 'accuracy': acc_score},
            ignore_index=True)
    plt.subplot(211)
    sns.pointplot(x=c_parameter_name, y="accuracy", data=df)

    title = 'Model Accuracy(%) vs ' + c_parameter_name + ' parameter'
    plt.title(title)
    plt.xticks(rotation=90)
    plt.grid()

    return df.loc[df['accuracy'].idxmax()][c_parameter_name]


def max_leaf_nodes_parameters(x_train, x_test, y_train, y_test):
    c_parameter_name = 'max_leaf_nodes'
    c_parameter_values = list(range(2, 100))
    df = pd.DataFrame(columns=[c_parameter_name, 'accuracy'])
    for input_parameter in c_parameter_values:
        clf = DecisionTreeClassifier(
            max_leaf_nodes=input_parameter, random_state=42)
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        df = df.append(
            {c_parameter_name: input_parameter, 'accuracy': acc_score},
            ignore_index=True)
    plt.subplot(211)
    sns.pointplot(x=c_parameter_name, y="accuracy", data=df)

    title = 'Model Accuracy(%) vs ' + c_parameter_name + ' parameter'
    plt.title(title)
    plt.xticks(rotation=90)
    plt.grid()

    return df.loc[df['accuracy'].idxmax()][c_parameter_name]


def min_impurity_decrease_parameters(x_train, x_test, y_train, y_test):
    c_parameter_name = 'min_impurity_decrease'
    c_parameter_values = [
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
        0.4]
    df = pd.DataFrame(columns=[c_parameter_name, 'accuracy'])
    for input_parameter in c_parameter_values:
        clf = DecisionTreeClassifier(
            min_impurity_decrease=input_parameter,
            random_state=42)
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        df = df.append(
            {c_parameter_name: input_parameter, 'accuracy': acc_score},
            ignore_index=True)
    plt.subplot(211)
    sns.pointplot(x=c_parameter_name, y="accuracy", data=df)
    plt.xticks(rotation=90)

    title = 'Model Accuracy(%) vs ' + c_parameter_name + ' parameter'
    plt.title(title)
    plt.grid()

    return df.loc[df['accuracy'].idxmax()][c_parameter_name]


def main():
    """Main function
    """
    df = load_data('diabetes.arff')
    features_cols = [i for i in df.columns.values.tolist() if i not in [
        'class']]
    x_train, x_test, y_train, y_test = train_test_split(
        df[features_cols], df['class'], test_size=0.2)
    clf = DecisionTreeClassifier(random_state=42)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred) * 100
    print(acc_score)
    max_depth = max_depth_parameters(x_train, x_test, y_train, y_test)
    print("Best max_depth t: ", max_depth)
    min_samples_split = int(min_samples_split_parameters(
        x_train, x_test, y_train, y_test))
    print("Best min samples split: ", min_samples_split)
    min_samples_leaf = int(min_samples_leaf_parameters(
        x_train, x_test, y_train, y_test))
    print("Best sample leaf: ", min_samples_leaf)
    max_leaf_nodes = int(max_leaf_nodes_parameters(
        x_train, x_test, y_train, y_test))
    print("Best max leaf nodes split: ", max_leaf_nodes)
    min_impurity_decrease = min_impurity_decrease_parameters(
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

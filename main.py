"""Main module for practical examination for data mining and knowledge discovery @ kent
"""

from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection._validation import validation_curve
from sklearn.utils import resample
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus

from find_best_parameters import find_best_parameters


def plot_balance_class(classes):
    """Plot balance between classes
    """
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
    data_frame = pd.DataFrame(data[0])
    return data_frame


def plot_validation_curve(X, Y, param_name, param_range):
    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(),
        X, Y, param_range=param_range, param_name=param_name, cv=5,
        scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(
        param_range,
        train_mean,
        color="darkorange",
        label="Training score")
    plt.plot(
        param_range,
        test_mean,
        label="Cross-validation score",
        color="navy")

    # Plot accurancy bands for training and test sets
    plt.fill_between(
        param_range,
        train_mean -
        train_std,
        train_mean +
        train_std,
        color="darkorange",
        alpha=0.2)
    plt.fill_between(
        param_range,
        test_mean -
        test_std,
        test_mean +
        test_std,
        color="navy",
        alpha=0.2)

    plt.title("Validation Curve With Decision Tree")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig(param_name + '.png')
    plt.clf()


def find_best_classifier(x_train, x_test, y_train, y_test):
    """Find best parameters for decision tree
    """
    max_depth, _ = find_best_parameters(
        'max_depth', list(range(1, 30)),
        x_train, x_test, y_train, y_test)
    print("Best max_depth t: ", max_depth)
    min_samples_split, _ = find_best_parameters(
        'min_samples_split', list(range(2, 400)),
        x_train, x_test, y_train, y_test)
    min_samples_split = int(min_samples_split)
    print("Best min samples split: ", min_samples_split)
    min_samples_leaf, _ = find_best_parameters(
        'min_samples_leaf', list(range(2, 200)),
        x_train, x_test, y_train, y_test)
    min_samples_leaf = int(min_samples_leaf)
    print("Best sample leaf: ", min_samples_leaf)
    max_leaf_nodes, _ = find_best_parameters(
        'max_leaf_nodes', list(range(2, 150)),
        x_train, x_test, y_train, y_test)
    max_leaf_nodes = int(max_leaf_nodes)
    print("Best max leaf nodes split: ", max_leaf_nodes)
    min_impurity_decrease, _ = find_best_parameters(
        'min_impurity_decrease', np.arange(0.0005, 0.1, 0.0005),
        x_train, x_test, y_train, y_test)
    print("Best min impurity decrease: ", min_impurity_decrease)
    clf = DecisionTreeClassifier(
        min_impurity_decrease=min_impurity_decrease,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_split=min_samples_split,
        random_state=0)
    clf = clf.fit(x_train, y_train)
    return clf


def search_with_grid(x_train, x_test, y_train, y_test):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(1, 10)) + [None],
        'min_samples_split': np.arange(10, 20),
        'min_samples_leaf': np.arange(10, 20),
        'max_leaf_nodes': np.arange(40, 70)
    }
    print("Commencing grid search")
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=0, max_leaf_nodes=58),
        param_grid,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy')
    grid.fit(x_train, y_train)
    clf = DecisionTreeClassifier(**grid.best_params_, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred) * 100
    print(
        "Accuracy after parameters tunning with grid search: accuracy: {0} | best_params {1} | best_score {2}".format(
            acc_score,
            grid.best_params_,
            grid.best_score_))
    return clf

def features_analysis(data_frame):
    tested_positive = data_frame[data_frame['class'] == 1]
    tested_positive.hist(figsize=(15, 10))
    plt.savefig('features_repartition_positive' + '.png')
    plt.clf()
    tested_negative = data_frame[data_frame['class'] == 0]
    tested_negative.hist(figsize=(15, 10))
    plt.savefig('features_repartition_negative' + '.png')
    plt.clf()
    scatter_matrix(data_frame, figsize=(15, 10))
    plt.savefig('correlation' + '.png')
    plt.clf()
    plt.hist(tested_negative['plas'], alpha=0.3)
    plt.hist(tested_positive['plas'], alpha=0.3)
    plt.xlabel("value of plas")
    plt.ylabel("number of subject")
    plt.savefig('comparaison_plas' + '.png')
    plt.clf()
    plt.hist(tested_negative['age'], alpha=0.3)
    plt.hist(tested_positive['age'], alpha=0.3)
    plt.xlabel("value of age")
    plt.ylabel("number of subject")
    plt.savefig('comparaison_age' + '.png')
    plt.clf()
    plt.hist(tested_negative['mass'], alpha=0.3)
    plt.hist(tested_positive['mass'], alpha=0.3)
    plt.xlabel("value of age")
    plt.ylabel("number of subject")
    plt.savefig('comparaison_mass' + '.png')
    plt.clf()

def vizu_validation_curve(X, Y):

    plot_validation_curve(X, Y, 'max_depth', np.arange(1, 30))
    plot_validation_curve(X, Y, 'min_samples_split', np.arange(2, 400))
    plot_validation_curve(X, Y, 'min_samples_leaf', np.arange(2, 125))
    plot_validation_curve(X, Y, 'max_leaf_nodes', np.arange(2, 200))
    plot_validation_curve(
        X, Y, 'min_impurity_decrease', np.arange(
            0.0005, 0.03, 0.0005))

def write_decision_tree(clf, name, features_names, class_names):
    dot_data = export_graphviz(clf, out_file=None, 
                                feature_names=features_names,  
                                class_names=class_names)
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_png(name + ".png")

def main():
    """Main function
    """
    data_frame = load_data('diabetes.arff')
    features_cols = [i for i in data_frame.columns.values.tolist() if i not in [
        'class']]
    #features_analysis(data_frame)
    X, Y = data_frame[features_cols], data_frame['class']
    vizu_validation_curve(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=0)
    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred) * 100
    write_decision_tree(clf, "raw", features_cols, ['negative', 'positive'])
    print("Accuracy before parameters tunning:", acc_score)


    clf = search_with_grid(x_train, x_test, y_train, y_test)
    y_pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred) * 100
    print(features_cols)
    print(list(clf.feature_importances_))
    write_decision_tree(clf, "grid", features_cols, ['negative', 'positive'])
    print("Accuracy after parameters tunning with search grid:", acc_score)


    clf = find_best_classifier(x_train, x_test, y_train, y_test)
    y_pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred) * 100
    print(features_cols)
    print(list(clf.feature_importances_))
    write_decision_tree(clf, "best_independent", features_cols, ['negative', 'positive'])
    print("Accuracy after parameters tunning with find best classifier:", acc_score)


if __name__ == "__main__":
    main()

"""SubModule for finding a best parameters for a decision tree
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def find_best_parameters(
        parameter_name,
        parameter_values,
        x_train,
        x_test,
        y_train,
        y_test):
    """Find best parameters
    """

    data_frame = pd.DataFrame(columns=[parameter_name, 'accuracy'])
    for parameter_value in parameter_values:
        clf = DecisionTreeClassifier(
            **{'random_state': 21, parameter_name: parameter_value})
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        data_frame = data_frame.append(
            {parameter_name: parameter_value, 'accuracy': acc_score},
            ignore_index=True)
    best = data_frame.loc[data_frame['accuracy'].idxmax()]
    return best[parameter_name], best['accuracy']

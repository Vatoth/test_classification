from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score


def find_best_parameters(
        parameter_name,
        parameter_values,
        x_train,
        x_test,
        y_train,
        y_test):

    df = pd.DataFrame(columns=[parameter_name, 'accuracy'])

    for parameter_value in parameter_values:
        clf = DecisionTreeClassifier(
            **{'random_state': 42, parameter_name: parameter_value})
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        df = df.append(
            {parameter_name: parameter_value, 'accuracy': acc_score},
            ignore_index=True)
    best = df.loc[df['accuracy'].idxmax()]
    return best[parameter_name], best['accuracy']

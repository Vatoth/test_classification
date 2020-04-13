from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score


def find_best_parameters(
        c_parameter_name,
        c_parameter_values,
        x_train,
        x_test,
        y_train,
        y_test):

    df = pd.DataFrame(columns=[c_parameter_name, 'accuracy'])

    for input_parameter in c_parameter_values:
        clf = DecisionTreeClassifier(
            **{'random_state': 42, c_parameter_name: input_parameter})
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        df = df.append(
            {c_parameter_name: input_parameter, 'accuracy': acc_score},
            ignore_index=True)
    best = df.loc[df['accuracy'].idxmax()]
    return best[c_parameter_name], best['accuracy']

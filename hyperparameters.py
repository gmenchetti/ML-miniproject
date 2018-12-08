from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

import numpy as np
import utils

balance = False
one_hot = False
X = None

print("Balance:", balance, "- one hot:", one_hot)
if balance:
    if one_hot:
        X = np.load("n_numpy_ds/balanced_one_hot_train.npy")
    else:
        X = np.load("n_numpy_ds/balanced_no_one_hot_train.npy")
else:
    if one_hot:
        X = np.load("n_numpy_ds/one_hot_train.npy")
    else:
        X = np.load("n_numpy_ds/no_one_hot_train.npy")
X_tr, y_tr = utils.get_Xy(X)

classifier = ["gnb", "mnb", "ridge", "svm", "dt", "gradb", "adab"]
scores = ['accuracy']#, 'precision', 'recall', 'f1']
for s in scores:
    print()
    print('SCORE:', s)
    print()
    for c in classifier:
        parameters = None
        model = None
        if c == "svm":
            parameters = {'kernel':('linear', 'rbf'), 'C':[ 0.01, 0.1, 1, 10]}
            model = SVC()
        elif c == "dt":
            parameters = {'max_depth': np.arange(1, 10, 1)}
            model = DecisionTreeClassifier()
        elif c == "gnb":
            parameters = {}
            model = GaussianNB()
        elif c == "mnb":
            parameters = {}
            model = MultinomialNB()
        elif c == "ridge":
            parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
            model = RidgeClassifier()
        elif c == "adab":
            est = np.arange(20, 520, 20)
            parameters = {'n_estimators': est}
            model = AdaBoostClassifier()
        elif c == "gradb":
            est = np.arange(20, 520, 20)
            parameters = {'n_estimators': est}
            model = GradientBoostingClassifier()

        clf = GridSearchCV(model, parameters, cv=5, scoring=s, verbose=0, return_train_score=True)
        clf.fit(X_tr, y_tr)

        print("Best parameters")
        print(clf.best_params_)
        print("")

        results = clf.cv_results_
        print("Classifier:", c)
        for i in range(0, len(results["mean_train_score"])):
            print(results["params"][i])
            print(s + " train:", results["mean_train_score"][i])
            print(s + " validation:", results["mean_test_score"][i])
            print()
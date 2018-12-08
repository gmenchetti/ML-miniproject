from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import numpy as np
import pickle
import utils
import sys

def main():
    X = None
    one_hot = False
    balanced = False

    if sys.argv[1] == "balance":
        balanced = True
        if sys.argv[2] == "one_hot":
            X = np.load("n_numpy_ds/balanced_one_hot_train.npy")
            one_hot = True
        elif sys.argv[2] == "no_one_hot":
            X = np.load("n_numpy_ds/balanced_no_one_hot_train.npy")
            one_hot = False
        else:
            print('Invalid second argument: one_hot - no_one_hot')
            exit(1)
    elif sys.argv[1] == "no_balance":
        balanced = False
        if sys.argv[2] == "one_hot":
            X = np.load("n_numpy_ds/one_hot_train.npy")
            one_hot = True
        elif sys.argv[2] == "no_one_hot":
            X = np.load("n_numpy_ds/no_one_hot_train.npy")
            one_hot = False
        else:
            print('Invalid second argument: one_hot - no_one_hot')
            exit(1)
    else:
        print('Invalid third argument: balance - no_balance')
        exit(1)

    X_tr, y_tr = utils.get_Xy(X)

    if balanced:
        if one_hot:
            svm_model = SVC(C=0.1, kernel='linear')
            svm_model.fit(X_tr, y_tr)
            with open("models/balanced_svm_one_hot.pkl", "wb") as output_file:
                pickle.dump(svm_model, output_file)

            ridge_model = RidgeClassifier(alpha=10)
            ridge_model.fit(X_tr, y_tr)
            with open("models/balanced_ridge_one_hot.pkl", "wb") as output_file:
                pickle.dump(ridge_model, output_file)

            gaus_nb_model = GaussianNB()
            gaus_nb_model.fit(X_tr, y_tr)
            with open("models/balanced_gausnb_one_hot.pkl", "wb") as output_file:
                pickle.dump(gaus_nb_model, output_file)

            multi_nb_model = MultinomialNB()
            multi_nb_model.fit(X_tr, y_tr)
            with open("models/balanced_multinb_one_hot.pkl", "wb") as output_file:
                pickle.dump(multi_nb_model, output_file)
        else:
            svm_model = SVC(C=1, kernel='linear')
            svm_model.fit(X_tr, y_tr)
            with open("models/balanced_svm_no_hot.pkl", "wb") as output_file:
                pickle.dump(svm_model, output_file)

            ridge_model = RidgeClassifier(alpha=0.1)
            ridge_model.fit(X_tr, y_tr)
            with open("models/balanced_ridge_no_hot.pkl", "wb") as output_file:
                pickle.dump(ridge_model, output_file)

            gaus_nb_model = GaussianNB()
            gaus_nb_model.fit(X_tr, y_tr)
            with open("models/balanced_gausnb_no_hot.pkl", "wb") as output_file:
                pickle.dump(gaus_nb_model, output_file)

            multi_nb_model = MultinomialNB()
            multi_nb_model.fit(X_tr, y_tr)
            with open("models/balanced_multinb_no_hot.pkl", "wb") as output_file:
                pickle.dump(multi_nb_model, output_file)
    else:
        if one_hot:
            svm_model = SVC(C=1, kernel='linear')
            svm_model.fit(X_tr, y_tr)
            with open("models/svm_one_hot.pkl", "wb") as output_file:
                pickle.dump(svm_model, output_file)

            ridge_model = RidgeClassifier(alpha=0.1)
            ridge_model.fit(X_tr, y_tr)
            with open("models/ridge_one_hot.pkl", "wb") as output_file:
                pickle.dump(ridge_model, output_file)

            gaus_nb_model = GaussianNB()
            gaus_nb_model.fit(X_tr, y_tr)
            with open("models/gausnb_one_hot.pkl", "wb") as output_file:
                pickle.dump(gaus_nb_model, output_file)

            multi_nb_model = MultinomialNB()
            multi_nb_model.fit(X_tr, y_tr)
            with open("models/multinb_one_hot.pkl", "wb") as output_file:
                pickle.dump(multi_nb_model, output_file)
        else:
            svm_model = SVC(C=1, kernel='rbf')
            svm_model.fit(X_tr, y_tr)
            with open("models/svm_no_hot.pkl", "wb") as output_file:
                pickle.dump(svm_model, output_file)

            ridge_model = RidgeClassifier(alpha=10)
            ridge_model.fit(X_tr, y_tr)
            with open("models/ridge_no_hot.pkl", "wb") as output_file:
                pickle.dump(ridge_model, output_file)

            gaus_nb_model = GaussianNB()
            gaus_nb_model.fit(X_tr, y_tr)
            with open("models/gausnb_no_hot.pkl", "wb") as output_file:
                pickle.dump(gaus_nb_model, output_file)

            multi_nb_model = MultinomialNB()
            multi_nb_model.fit(X_tr, y_tr)
            with open("models/multinb_no_hot.pkl", "wb") as output_file:
                pickle.dump(multi_nb_model, output_file)

if __name__ == '__main__':
    main()
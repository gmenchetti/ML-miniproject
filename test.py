from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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
            X = np.load("n_numpy_ds/balanced_one_hot_test.npy")
            one_hot = True
        elif sys.argv[2] == "no_one_hot":
            X = np.load("n_numpy_ds/balanced_no_one_hot_test.npy")
            one_hot = False
        else:
            print('Invalid second argument: one_hot - no_one_hot')
            exit(1)
    elif sys.argv[1] == "no_balance":
        balanced = False
        if sys.argv[2] == "one_hot":
            X = np.load("n_numpy_ds/one_hot_test.npy")
            one_hot = True
        elif sys.argv[2] == "no_one_hot":
            X = np.load("n_numpy_ds/no_one_hot_test.npy")
            one_hot = False
        else:
            print('Invalid second argument: one_hot - no_one_hot')
            exit(1)
    else:
        print('Invalid third argument: balance - no_balance')
        exit(1)
    X_te, y_te = utils.get_Xy(X)

    print("BASELINE most frequent")
    unique, counts = np.unique(y_te, return_counts=True)
    most_freq_label = unique[np.argmax(counts)]
    baseline_pred = np.full((len(y_te), 1), most_freq_label)
    print("Acc:", accuracy_score(y_te, baseline_pred))
    print("Prec:", precision_score(y_te, baseline_pred))
    print("Rec:", recall_score(y_te, baseline_pred))
    print("f1:", f1_score(y_te, baseline_pred))
    print()
    if balanced:
        filname = "models/balanced_"
        print("Balanced dataset")
        if one_hot:
            print("Results with One Hot Encoding")
            print()
            with open(filname+"svm_one_hot.pkl", "rb") as file:
                print("SVM")
                svm_model = pickle.load(file)
                preds = svm_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open(filname+"ridge_one_hot.pkl", "rb") as file:
                print("Ridge Classifier")
                ridge_model = pickle.load(file)
                preds = ridge_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open(filname+"gausnb_one_hot.pkl", "rb") as file:
                print("Gaussian Naive Bayes Classifier")
                gaus_nb_model = pickle.load(file)
                preds = gaus_nb_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open(filname+"multinb_one_hot.pkl", "rb") as file:
                print("Multinomial Naive Bayes Classifier")
                multi_nb_model = pickle.load(file)
                preds = multi_nb_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
        else:
            print("Results without One Hot Encoding")
            print()
            with open(filname+"svm_no_hot.pkl", "rb") as file:
                print("SVM")
                svm_model = pickle.load(file)
                preds = svm_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open(filname+"ridge_no_hot.pkl", "rb") as file:
                print("Ridge Classifier")
                ridge_model = pickle.load(file)
                preds = ridge_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open(filname+"gausnb_no_hot.pkl", "rb") as file:
                print("Gaussian Naive Bayes Classifier")
                gaus_nb_model = pickle.load(file)
                preds = gaus_nb_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open(filname+"multinb_no_hot.pkl", "rb") as file:
                print("Multinomial Naive Bayes Classifier")
                multi_nb_model = pickle.load(file)
                preds = multi_nb_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
    else:
        print("Imbalanced dataset")
        if one_hot:
            print("Results with One Hot Encoding")
            print()
            with open("models/svm_one_hot.pkl", "rb") as file:
                print("SVM")
                svm_model = pickle.load(file)
                preds = svm_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open("models/ridge_one_hot.pkl", "rb") as file:
                print("Ridge Classifier")
                ridge_model = pickle.load(file)
                preds = ridge_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open("models/gausnb_one_hot.pkl", "rb") as file:
                print("Gaussian Naive Bayes Classifier")
                gaus_nb_model = pickle.load(file)
                preds = gaus_nb_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open("models/multinb_one_hot.pkl", "rb") as file:
                print("Multinomial Naive Bayes Classifier")
                multi_nb_model = pickle.load(file)
                preds = multi_nb_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
        else:
            print("Results without One Hot Encoding")
            print()
            with open("models/svm_no_hot.pkl", "rb") as file:
                print("SVM")
                svm_model = pickle.load(file)
                preds = svm_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open("models/ridge_no_hot.pkl", "rb") as file:
                print("Ridge Classifier")
                ridge_model = pickle.load(file)
                preds = ridge_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open("models/gausnb_no_hot.pkl", "rb") as file:
                print("Gaussian Naive Bayes Classifier")
                gaus_nb_model = pickle.load(file)
                preds = gaus_nb_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))
            print()
            with open("models/multinb_no_hot.pkl", "rb") as file:
                print("Multinomial Naive Bayes Classifier")
                multi_nb_model = pickle.load(file)
                preds = multi_nb_model.predict(X_te)
                print("Acc:", accuracy_score(y_te, preds))
                print("Prec:", precision_score(y_te, preds))
                print("Rec:", recall_score(y_te, preds))
                print("f1:", f1_score(y_te, preds))


if __name__ == '__main__':
    main()
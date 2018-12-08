from sklearn.model_selection import train_test_split
from preprocessing import Preprocessor
import features_selection as fs
import pandas as pd
import numpy as np


def preprocess_and_split(balanced=False, one_hot=False):
    df = pd.read_csv("data/responses.csv")
    preprocessor = Preprocessor(df)
    preprocessor.create_labels(balanced)
    preprocessor.apply(one_hot)
    X, y = preprocessor.get_Xy_df()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    return X_train, X_test, y_train, y_test


def create_datasets(X_tr, X_te, y_tr, y_te, balance, one_hot, th):
    f_selector = fs.select_from_model(X_tr, y_tr, th)
    X_tr = f_selector.transform(X_tr)
    X_te = f_selector.transform(X_te)

    test_complete = np.concatenate((X_te, y_te.reshape(-1, 1)), axis=1)
    train_complete = np.concatenate((X_tr, y_tr.reshape(-1, 1)), axis=1)

    if balance:
        if one_hot:
            np.save("n_numpy_ds/balanced_one_hot_test", test_complete)
            np.save("n_numpy_ds/balanced_one_hot_train", train_complete)
        else:
            np.save("n_numpy_ds/balanced_no_one_hot_test", test_complete)
            np.save("n_numpy_ds/balanced_no_one_hot_train", train_complete)
    else:
        if one_hot:
            np.save("n_numpy_ds/one_hot_test", test_complete)
            np.save("n_numpy_ds/one_hot_train", train_complete)
        else:
            np.save("n_numpy_ds/no_one_hot_test", test_complete)
            np.save("n_numpy_ds/no_one_hot_train", train_complete)


def split_features_labels(X):
    y = X["Empathy"]
    X = X.drop(["Empathy"], axis=1)
    return X.as_matrix(), y.as_matrix()


def get_Xy(X):
    return X[:, :-1], X[:, -1]

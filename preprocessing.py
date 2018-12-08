from sklearn.utils import resample
from sklearn import preprocessing
import pandas as pd
import numpy as np


class Preprocessor():
    def __init__(self, df=None):
        self.df = df
        self.X = None
        self.y = None
        self.Xfinal = None
        self.yfinal = None
        self.feature_selection = False

    def create_labels(self, balance=False):
        cleared_labels = self.df[self.df["Empathy"].notnull()]
        if balance:
            cleared_labels = self.balance_ds(self.df)
        self.y = cleared_labels["Empathy"]
        self.X = cleared_labels.drop(["Empathy"], axis=1)

    def apply(self, one_hot=False):
        self.object_to_categories()
        self.impute_median()
        self.transform_to_int()
        self.remove_outliers()
        if one_hot:
            self.one_hot_encode_features()
        self.one_hot_encode_labels()

    def object_to_categories(self):
        names = [key for key in dict(self.X.dtypes) if dict(self.X.dtypes)[key] in ['object']]
        for n in names:
            self.X[n] = self.X[n].astype('category')
            self.X[n] = self.X[n].cat.codes
            self.X[n] = self.X[n].replace({-1:np.nan})

    def impute_median(self):
        median = self.X.median()
        columns = list(self.X.columns.values)
        for m,c in zip(median, columns):
            self.X[c] = self.X[c].fillna(m)

    def transform_to_int(self):
        names = [key for key in dict(self.X.dtypes) if dict(self.X.dtypes)[key] in ['float64']]
        for n in names:
            self.X[n] = self.X[n].astype('int64')

    def remove_outliers(self):
        Q1 = self.X['Weight'].quantile(0.25)
        Q2 = self.X['Weight'].quantile(0.85)
        self.X.loc[self.X['Weight'] < Q1, 'Weight'] = Q1
        self.X.loc[self.X['Weight'] > Q2, 'Weight'] = Q2

        Q1 = self.X['Height'].quantile(0.25)
        Q2 = self.X['Height'].quantile(0.85)
        self.X.loc[self.X['Height'] < Q1, 'Height'] = Q1
        self.X.loc[self.X['Height'] > Q2, 'Height'] = Q2

    def one_hot_encode_labels(self):
        print(self.y.value_counts())
        self.y = self.y.replace([1, 2, 3], [0, 0, 0])
        self.y = self.y.replace([4, 5], [1, 1])

    def encode_age_interval(self):
        new_age = []
        for i, r in (self.X["Age"].to_frame()).iterrows():
            if r["Age"] < 18:
                new_age.append(0)
            elif (r["Age"] >= 18 and r["Age"] < 23):
                new_age.append(1)
            elif (r["Age"] >= 23 and r["Age"] < 26):
                new_age.append(2)
            else:
                new_age.append(3)
        new_age = np.array(new_age).reshape(-1, 1)
        enc = preprocessing.OneHotEncoder()
        enc.fit(new_age)
        onehot = enc.transform(new_age).toarray()
        return onehot

    def balance_ds(self, df):
        counts = df["Empathy"].value_counts()
        labels = counts.index.tolist()
        df_minority_1 = df[df["Empathy"] == labels[2]]
        df_minority_2 = df[df["Empathy"] == labels[3]]
        df_minority_3 = df[df["Empathy"] == labels[4]]
        df_majority_1 = df[df["Empathy"] == labels[0]]
        df_majority_2 = df[df["Empathy"] == labels[1]]

        print("Downsampling DS")
        number = int(round(np.sum([len(df_minority_1), len(df_minority_2), len(df_minority_3)])/2))
        df_majority_downsampled_1 = resample(df_majority_1,
                                           replace=False,  # sample without replacement
                                           n_samples=number)

        df_majority_downsampled_2 = resample(df_majority_2,
                                           replace=False,  # sample without replacement
                                           n_samples=number)
        df_downsampled = pd.concat(
            [df_majority_downsampled_1, df_majority_downsampled_2, df_minority_1, df_minority_2, df_minority_3])

        print(df_downsampled["Empathy"].value_counts())
        return df_downsampled

    def one_hot_encode_features(self):
        newX = None
        if "Age" in self.X.columns.values:
            newX = self.X.drop(["Age"], axis=1, inplace=False)
        if "Weight" in self.X.columns.values:
            newX = newX.drop(["Weight"], axis=1, inplace=False)
        if "Height" in self.X.columns.values:
            newX = newX.drop(["Height"], axis=1, inplace=False)
        if newX is None:
            newX = self.X
        enc = preprocessing.OneHotEncoder()
        enc.fit(newX)
        onehot = enc.transform(newX).toarray()
        final = onehot
        if "Age" in self.X.columns.values:
            age = self.encode_age_interval()
            final = np.hstack((final, age))
        if "Weight" in self.X.columns.values:
            weight = self.X["Weight"].as_matrix().reshape(-1, 1)
            final = np.hstack((final, weight))
        if "Height" in self.X.columns.values:
            height = self.X["Height"].as_matrix().reshape(-1, 1)
            final = np.hstack((final, height))
        self.Xfinal = final

    def get_Xy_matrix(self):
        if self.Xfinal is None:
            self.Xfinal = self.X
        if self.yfinal is None:
            self.yfinal = self.y
        if type(self.Xfinal) is not np.ndarray:
            self.Xfinal = self.Xfinal.values
        if type(self.yfinal) is not np.ndarray:
            self.yfinal = self.yfinal.values
        return self.Xfinal, self.yfinal

    def get_Xy_df(self):
        if self.Xfinal is None:
            self.Xfinal = self.X
        if self.yfinal is None:
            self.yfinal = self.y
        return self.Xfinal, self.yfinal

    def set_Xy(self, X, y):
        self.X = X
        self.y = y

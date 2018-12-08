from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC

def variance_fs(X, th):
    sel = VarianceThreshold(threshold=th)
    var_fs = sel.fit(X)
    return var_fs

def select_k_best(X, y, k):
    model = SelectKBest(chi2, k=k).fit(X,y)
    return model

def apply_PCA(X, n_components):
    pca = PCA(n_components)
    pca.fit(X)
    return pca

def select_from_model(X, y, th):
    y = y.reshape(-1, 1)
    #clf = LinearSVC(C=1, penalty="l1", dual=False)
    # clf = Lasso(alpha=0.01)
    clf = ExtraTreesClassifier(n_estimators=100)
    model = None
    if th is None:
        model = SelectFromModel(clf)
    else:
        model = SelectFromModel(clf, threshold=th)
    model.fit(X, y)
    return model
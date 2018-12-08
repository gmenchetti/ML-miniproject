import utils

one_hot = True
balance = False

X_tr, X_te, y_tr, y_te = utils.preprocess_and_split(balance, one_hot)

th = None
if not balance:
    th = 0
    if one_hot:
        th = 0.002
    else:
        th = 0.007

utils.create_datasets(X_tr, X_te, y_tr, y_te, balance=balance, one_hot=one_hot, th=th)

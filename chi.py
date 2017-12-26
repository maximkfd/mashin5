import numpy as np

from scipy import special
from scipy.sparse import spmatrix


def _chisquare(f_obs, f_exp):
    f_obs = np.asarray(f_obs, dtype=np.float64)

    k = len(f_obs)
    # Reuse f_obs for chi-squared statistics
    chisq = f_obs
    chisq -= f_exp
    chisq **= 2
    with np.errstate(invalid="ignore"):
        chisq /= f_exp
    chisq = chisq.sum(axis=0)
    return chisq, special.chdtrc(k - 1, chisq)


def safe_sparse_dot(a, b, dense_output=False):
    if isinstance(a, spmatrix) or isinstance(b, spmatrix):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


def chi2(X, y):
    Y = np.vstack((y + 1) / 2)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(Y.T, X)

    feature_count = X.sum(axis=0).reshape(1, -1)
    class_prob = Y.mean(axis=0).reshape(1, -1)
    expected = np.dot(class_prob.T, feature_count)

    return _chisquare(observed, expected)

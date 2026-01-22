from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp
def condentropy(px, py, pxpy=None, logbase=2):
    """
    Return the conditional entropy of X given Y.

    Parameters
    ----------
    px : array_like
    py : array_like
    pxpy : array_like, optional
        If pxpy is None, the distributions are assumed to be independent
        and conendtropy(px,py) = shannonentropy(px)
    logbase : int or np.e

    Returns
    -------
    sum_{kj}log(q_{j}/w_{kj}

    where q_{j} = Y[j]
    and w_kj = X[k,j]
    """
    if not _isproperdist(px) or not _isproperdist(py):
        raise ValueError('px or py is not a proper probability distribution')
    if pxpy is not None and (not _isproperdist(pxpy)):
        raise ValueError('pxpy is not a proper joint distribtion')
    if pxpy is None:
        pxpy = np.outer(py, px)
    condent = np.sum(pxpy * np.nan_to_num(np.log2(py / pxpy)))
    if logbase == 2:
        return condent
    else:
        return logbasechange(2, logbase) * condent
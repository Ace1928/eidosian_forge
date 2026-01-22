from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp
def shannoninfo(px, logbase=2):
    """
    Shannon's information

    Parameters
    ----------
    px : float or array_like
        `px` is a discrete probability distribution

    Returns
    -------
    For logbase = 2
    np.log2(px)
    """
    px = np.asarray(px)
    if not np.all(px <= 1) or not np.all(px >= 0):
        raise ValueError('px does not define proper distribution')
    if logbase != 2:
        return -logbasechange(2, logbase) * np.log2(px)
    else:
        return -np.log2(px)
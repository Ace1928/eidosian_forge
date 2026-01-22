from statsmodels.compat.python import lrange
import numpy as np
from scipy import ndimage
def groupstatsbin(factors, values):
    """uses np.bincount, assumes factors/labels are integers
    """
    n = len(factors)
    ix, rind = np.unique(factors, return_inverse=1)
    gcount = np.bincount(rind)
    gmean = np.bincount(rind, weights=values) / (1.0 * gcount)
    meanarr = gmean[rind]
    withinvar = np.bincount(rind, weights=(values - meanarr) ** 2) / (1.0 * gcount)
    withinvararr = withinvar[rind]
    return (gcount, gmean, meanarr, withinvar, withinvararr)
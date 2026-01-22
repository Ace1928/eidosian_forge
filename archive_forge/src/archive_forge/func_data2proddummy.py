import numpy as np
import numpy.lib.recfunctions
from statsmodels.compat.python import lmap
from statsmodels.regression.linear_model import OLS
def data2proddummy(x):
    """creates product dummy variables from 2 columns of 2d array

    drops last dummy variable, but not from each category
    singular with simple dummy variable but not with constant

    quickly written, no safeguards

    """
    groups = np.unique(lmap(tuple, x.tolist()))
    return (x == groups[:, None, :]).all(-1).T.astype(int)[:, :-1]
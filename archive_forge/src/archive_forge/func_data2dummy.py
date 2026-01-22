import numpy as np
import numpy.lib.recfunctions
from statsmodels.compat.python import lmap
from statsmodels.regression.linear_model import OLS
def data2dummy(x, returnall=False):
    """convert array of categories to dummy variables
    by default drops dummy variable for last category
    uses ravel, 1d only"""
    x = x.ravel()
    groups = np.unique(x)
    if returnall:
        return (x[:, None] == groups).astype(int)
    else:
        return (x[:, None] == groups).astype(int)[:, :-1]
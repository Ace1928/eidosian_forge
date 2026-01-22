import numpy as np
import numpy.lib.recfunctions
from statsmodels.compat.python import lmap
from statsmodels.regression.linear_model import OLS
def data2groupcont(x1, x2):
    """create dummy continuous variable

    Parameters
    ----------
    x1 : 1d array
        label or group array
    x2 : 1d array (float)
        continuous variable

    Notes
    -----
    useful for group specific slope coefficients in regression
    """
    if x2.ndim == 1:
        x2 = x2[:, None]
    dummy = data2dummy(x1, returnall=True)
    return dummy * x2
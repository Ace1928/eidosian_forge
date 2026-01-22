from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def d_or_f(x, width=6):
    """convert number to string with either integer of float formatting

    This is used internally for nobs and degrees of freedom which are usually
    integers but can be float in some cases.

    Parameters
    ----------
    x : int or float
    width : int
        only used if x is nan

    Returns
    -------
    str : str
        number as formatted string
    """
    if np.isnan(x):
        return (width - 3) * ' ' + 'NaN'
    if x // 1 == x:
        return '%#6d' % x
    else:
        return '%#8.2f' % x
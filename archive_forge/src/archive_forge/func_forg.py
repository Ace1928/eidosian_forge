from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def forg(x, prec=3):
    x = np.squeeze(x)
    if prec == 3:
        if abs(x) >= 10000.0 or abs(x) < 0.0001:
            return '%9.3g' % x
        else:
            return '%9.3f' % x
    elif prec == 4:
        if abs(x) >= 10000.0 or abs(x) < 0.0001:
            return '%10.4g' % x
        else:
            return '%10.4f' % x
    else:
        raise ValueError('`prec` argument must be either 3 or 4, not {prec}'.format(prec=prec))
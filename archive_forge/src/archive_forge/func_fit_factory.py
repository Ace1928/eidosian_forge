from __future__ import (absolute_import, division, print_function)
import warnings
from math import exp
import numpy as np
def fit_factory(discard=1):

    def fit(x, y):
        p = np.polyfit(x, y, 1)
        v = np.polyval(p, x)
        e = np.abs(y - v)
        drop_idxs = np.argsort(e)[-discard]
        return np.polyfit(np.delete(x, drop_idxs), np.delete(y, drop_idxs), 1)
    return fit
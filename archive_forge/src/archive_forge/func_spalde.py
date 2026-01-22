import warnings
import numpy as np
from . import _fitpack
from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose,
from . import dfitpack
def spalde(x, tck):
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, x=x, t=t, k=k: spalde(x, [t, c, k]), c))
    else:
        x = atleast_1d(x)
        if len(x) > 1:
            return list(map(lambda x, tck=tck: spalde(x, tck), x))
        d, ier = dfitpack.spalde(t, c, k + 1, x[0])
        if ier == 0:
            return d
        if ier == 10:
            raise TypeError('Invalid input data. t(k)<=x<=t(n-k+1) must hold.')
        raise TypeError('Unknown error')
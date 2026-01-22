import warnings
import numpy as np
from . import _fitpack
from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose,
from . import dfitpack
def splev(x, tck, der=0, ext=0):
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, x=x, t=t, k=k, der=der: splev(x, [t, c, k], der, ext), c))
    else:
        if not 0 <= der <= k:
            raise ValueError('0<=der=%d<=k=%d must hold' % (der, k))
        if ext not in (0, 1, 2, 3):
            raise ValueError('ext = %s not in (0, 1, 2, 3) ' % ext)
        x = asarray(x)
        shape = x.shape
        x = atleast_1d(x).ravel()
        if der == 0:
            y, ier = dfitpack.splev(t, c, k, x, ext)
        else:
            y, ier = dfitpack.splder(t, c, k, x, der, ext)
        if ier == 10:
            raise ValueError('Invalid input data')
        if ier == 1:
            raise ValueError('Found x value not in the domain')
        if ier:
            raise TypeError('An error occurred')
        return y.reshape(shape)
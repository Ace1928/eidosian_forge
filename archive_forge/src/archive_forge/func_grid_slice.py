from __future__ import print_function
import os
import io
import time
import functools
import collections
import collections.abc
import numpy as np
import requests
import IPython
def grid_slice(amin, amax, shape, bmin, bmax):
    """Give a slice such that [amin, amax] is in [bmin, bmax].

    Given a grid with shape, and begin and end coordinates amin, amax, what slice
    do we need to take such that it minimally covers bmin, bmax.

    amin, amax = 0, 1; shape = 4
    0  0.25  0.5  0.75  1
    |    |    |    |    |
    bmin, bmax = 0.5, 1.0 should give 2,4, 0.5, 1.0
    bmin, bmax = 0.4, 1.0 should give 1,4, 0.25, 1.0

    bmin, bmax = -1, 1.0 should give 0,4, 0, 1.0

    what about negative bmin and bmax ?
    It will just flip bmin and bmax
    bmin, bmax = 1.0, 0.5 should give 2,4, 0.5, 1.5

    amin, amax = 1, 0; shape = 4
    1  0.75  0.5  0.25  0
    |    |    |    |    |
    bmin, bmax = 0.5, 1.0 should give 0,2, 1.0, 0.5
    bmin, bmax = 0.4, 1.0 should give 0,3, 1.0, 0.25
    """
    width = amax - amin
    bmin, bmax = (min(bmin, bmax), max(bmin, bmax))
    nmin = (bmin - amin) / width
    nmax = (bmax - amin) / width
    if width < 0:
        imin = max(0, int(np.floor(nmax * shape)))
        imax = min(shape, int(np.ceil(nmin * shape)))
    else:
        imin = max(0, int(np.floor(nmin * shape)))
        imax = min(shape, int(np.ceil(nmax * shape)))
    nmin = imin / shape
    nmax = imax / shape
    return ((imin, imax), (amin + nmin * width, amin + nmax * width))
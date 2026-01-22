import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def load_testing_files():
    for fn in _filenames:
        name = fn.replace('.txt', '').replace('-ml', '')
        fqfn = os.path.join(os.path.dirname(__file__), 'data', fn)
        fp = open(fqfn)
        eo[name] = np.loadtxt(fp)
        fp.close()
    eo['pdist-boolean-inp'] = np.bool_(eo['pdist-boolean-inp'])
    eo['random-bool-data'] = np.bool_(eo['random-bool-data'])
    eo['random-float32-data'] = np.float32(eo['random-double-data'])
    eo['random-int-data'] = np_long(eo['random-int-data'])
    eo['random-uint-data'] = np_ulong(eo['random-uint-data'])
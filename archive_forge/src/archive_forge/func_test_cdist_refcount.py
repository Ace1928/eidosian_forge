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
def test_cdist_refcount(self, metric):
    x1 = np.random.rand(10, 10)
    x2 = np.random.rand(10, 10)
    kwargs = dict()
    if metric == 'minkowski':
        kwargs['p'] = 1.23
    out = cdist(x1, x2, metric=metric, **kwargs)
    weak_refs = [weakref.ref(v) for v in (x1, x2, out)]
    del x1, x2, out
    if IS_PYPY:
        break_cycles()
    assert all((weak_ref() is None for weak_ref in weak_refs))
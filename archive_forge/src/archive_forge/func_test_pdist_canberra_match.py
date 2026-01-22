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
@pytest.mark.slow
def test_pdist_canberra_match(self):
    D = eo['iris']
    if verbose > 2:
        print(D.shape, D.dtype)
    eps = 1e-15
    y1 = wpdist_no_const(D, 'canberra')
    y2 = wpdist_no_const(D, 'test_canberra')
    assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)
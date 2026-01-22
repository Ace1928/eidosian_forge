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
def test_pdist_jensenshannon_iris_nonC(self):
    eps = 5e-05
    X = eo['iris']
    Y_right = eo['pdist-jensenshannon-iris']
    Y_test2 = pdist(X, 'test_jensenshannon')
    assert_allclose(Y_test2, Y_right, rtol=eps)
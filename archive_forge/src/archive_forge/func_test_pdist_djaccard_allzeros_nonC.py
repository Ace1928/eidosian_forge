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
def test_pdist_djaccard_allzeros_nonC(self):
    eps = 1e-15
    Y = pdist(np.zeros((5, 3)), 'test_jaccard')
    assert_allclose(np.zeros(10), Y, rtol=eps)
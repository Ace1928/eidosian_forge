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
def test_immutable_input(metric):
    if metric in ('jensenshannon', 'mahalanobis', 'seuclidean'):
        pytest.skip('not applicable')
    x = np.arange(10, dtype=np.float64)
    x.setflags(write=False)
    getattr(scipy.spatial.distance, metric)(x, x, w=x)
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
def test_cosine(self):
    for x, y in self.cases:
        dist = wcosine(x, y)
        assert_almost_equal(dist, 1.0 - 18.0 / (np.sqrt(14) * np.sqrt(27)))
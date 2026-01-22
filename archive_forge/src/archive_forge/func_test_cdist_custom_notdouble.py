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
def test_cdist_custom_notdouble(self):

    class myclass:
        pass

    def _my_metric(x, y):
        if not isinstance(x[0], myclass) or not isinstance(y[0], myclass):
            raise ValueError('Type has been changed')
        return 1.123
    data = np.array([[myclass()]], dtype=object)
    cdist_y = cdist(data, data, metric=_my_metric)
    right_y = 1.123
    assert_equal(cdist_y, right_y, verbose=verbose > 2)
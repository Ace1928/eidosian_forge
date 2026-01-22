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
def test_cdist_extra_args(self, metric):
    X1 = [[1.0, 2.0, 3.0], [1.2, 2.3, 3.4], [2.2, 2.3, 4.4]]
    X2 = [[7.0, 5.0, 8.0], [7.5, 5.8, 8.4], [5.5, 5.8, 4.4]]
    kwargs = {'N0tV4l1D_p4raM': 3.14, 'w': np.arange(3)}
    args = [3.14] * 200
    with pytest.raises(TypeError):
        cdist(X1, X2, metric=metric, **kwargs)
    with pytest.raises(TypeError):
        cdist(X1, X2, metric=eval(metric), **kwargs)
    with pytest.raises(TypeError):
        cdist(X1, X2, metric='test_' + metric, **kwargs)
    with pytest.raises(TypeError):
        cdist(X1, X2, *args, metric=metric)
    with pytest.raises(TypeError):
        cdist(X1, X2, *args, metric=eval(metric))
    with pytest.raises(TypeError):
        cdist(X1, X2, *args, metric='test_' + metric)
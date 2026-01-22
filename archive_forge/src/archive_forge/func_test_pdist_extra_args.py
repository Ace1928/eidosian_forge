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
def test_pdist_extra_args(self, metric):
    X1 = [[1.0, 2.0], [1.2, 2.3], [2.2, 2.3]]
    kwargs = {'N0tV4l1D_p4raM': 3.14, 'w': np.arange(2)}
    args = [3.14] * 200
    with pytest.raises(TypeError):
        pdist(X1, metric=metric, **kwargs)
    with pytest.raises(TypeError):
        pdist(X1, metric=eval(metric), **kwargs)
    with pytest.raises(TypeError):
        pdist(X1, metric='test_' + metric, **kwargs)
    with pytest.raises(TypeError):
        pdist(X1, *args, metric=metric)
    with pytest.raises(TypeError):
        pdist(X1, *args, metric=eval(metric))
    with pytest.raises(TypeError):
        pdist(X1, *args, metric='test_' + metric)
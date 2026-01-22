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
def test_Xdist_non_negative_weights(metric):
    X = eo['random-float32-data'][::5, ::2]
    w = np.ones(X.shape[1])
    w[::5] = -w[::5]
    if metric in ['seuclidean', 'mahalanobis', 'jensenshannon']:
        pytest.skip('not applicable')
    for m in [metric, eval(metric), 'test_' + metric]:
        with pytest.raises(ValueError):
            pdist(X, m, w=w)
        with pytest.raises(ValueError):
            cdist(X, X, m, w=w)
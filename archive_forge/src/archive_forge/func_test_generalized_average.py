import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_generalized_average():
    a, b = (1, 2)
    methods = ['min', 'geometric', 'arithmetic', 'max']
    means = [_generalized_average(a, b, method) for method in methods]
    assert means[0] <= means[1] <= means[2] <= means[3]
    c, d = (12, 12)
    means = [_generalized_average(c, d, method) for method in methods]
    assert means[0] == means[1] == means[2] == means[3]
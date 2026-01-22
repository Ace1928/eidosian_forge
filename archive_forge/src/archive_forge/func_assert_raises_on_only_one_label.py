import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
def assert_raises_on_only_one_label(func):
    """Assert message when there is only one label"""
    rng = np.random.RandomState(seed=0)
    with pytest.raises(ValueError, match='Number of labels is'):
        func(rng.rand(10, 2), np.zeros(10))
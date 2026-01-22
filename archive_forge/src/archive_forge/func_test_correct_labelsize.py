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
def test_correct_labelsize():
    dataset = datasets.load_iris()
    X = dataset.data
    y = np.arange(X.shape[0])
    err_msg = 'Number of labels is %d\\. Valid values are 2 to n_samples - 1 \\(inclusive\\)' % len(np.unique(y))
    with pytest.raises(ValueError, match=err_msg):
        silhouette_score(X, y)
    y = np.zeros(X.shape[0])
    err_msg = 'Number of labels is %d\\. Valid values are 2 to n_samples - 1 \\(inclusive\\)' % len(np.unique(y))
    with pytest.raises(ValueError, match=err_msg):
        silhouette_score(X, y)
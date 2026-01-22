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
def test_silhouette_score_integer_precomputed():
    """Check that silhouette_score works for precomputed metrics that are integers.

    Non-regression test for #22107.
    """
    result = silhouette_score([[0, 1, 2], [1, 0, 1], [2, 1, 0]], [0, 0, 1], metric='precomputed')
    assert result == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match='contains non-zero'):
        silhouette_score([[1, 1, 2], [1, 0, 1], [2, 1, 0]], [0, 0, 1], metric='precomputed')
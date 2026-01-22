import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
@pytest.mark.parametrize('labels_true, labels_pred', [(['a'] * 6, [1, 1, 0, 0, 1, 1]), ([1] * 6, [1, 1, 0, 0, 1, 1]), ([1, 1, 0, 0, 1, 1], ['a'] * 6), ([1, 1, 0, 0, 1, 1], [1] * 6), (['a'] * 6, ['a'] * 6)])
def test_mutual_info_score_positive_constant_label(labels_true, labels_pred):
    assert mutual_info_score(labels_true, labels_pred) == 0
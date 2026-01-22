import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_perfect_matches():
    for score_func in score_funcs:
        assert score_func([], []) == pytest.approx(1.0)
        assert score_func([0], [1]) == pytest.approx(1.0)
        assert score_func([0, 0, 0], [0, 0, 0]) == pytest.approx(1.0)
        assert score_func([0, 1, 0], [42, 7, 42]) == pytest.approx(1.0)
        assert score_func([0.0, 1.0, 0.0], [42.0, 7.0, 42.0]) == pytest.approx(1.0)
        assert score_func([0.0, 1.0, 2.0], [42.0, 7.0, 2.0]) == pytest.approx(1.0)
        assert score_func([0, 1, 2], [42, 7, 2]) == pytest.approx(1.0)
    score_funcs_with_changing_means = [normalized_mutual_info_score, adjusted_mutual_info_score]
    means = {'min', 'geometric', 'arithmetic', 'max'}
    for score_func in score_funcs_with_changing_means:
        for mean in means:
            assert score_func([], [], average_method=mean) == pytest.approx(1.0)
            assert score_func([0], [1], average_method=mean) == pytest.approx(1.0)
            assert score_func([0, 0, 0], [0, 0, 0], average_method=mean) == pytest.approx(1.0)
            assert score_func([0, 1, 0], [42, 7, 42], average_method=mean) == pytest.approx(1.0)
            assert score_func([0.0, 1.0, 0.0], [42.0, 7.0, 42.0], average_method=mean) == pytest.approx(1.0)
            assert score_func([0.0, 1.0, 2.0], [42.0, 7.0, 2.0], average_method=mean) == pytest.approx(1.0)
            assert score_func([0, 1, 2], [42, 7, 2], average_method=mean) == pytest.approx(1.0)
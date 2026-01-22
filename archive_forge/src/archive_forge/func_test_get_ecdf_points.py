import pytest
import numpy as np
import scipy.stats
from ...stats.ecdf_utils import (
@pytest.mark.parametrize('difference', [True, False])
def test_get_ecdf_points(difference):
    """Test _get_ecdf_points."""
    sample = np.array([1, 2, 3, 3, 4, 5, 5])
    eval_points = np.arange(-1, 7, 0.1)
    x, y = _get_ecdf_points(sample, eval_points, difference)
    assert np.array_equal(x, eval_points)
    assert np.array_equal(y, compute_ecdf(sample, eval_points))
    eval_points = np.arange(1, 6, 0.1)
    x, y = _get_ecdf_points(sample, eval_points, difference)
    assert len(x) == len(eval_points) + 1 - difference
    assert len(y) == len(eval_points) + 1 - difference
    if not difference:
        assert x[0] == eval_points[0]
        assert y[0] == 0
        assert np.allclose(x[1:], eval_points)
        assert np.allclose(y[1:], compute_ecdf(sample, eval_points))
        assert x[-1] == eval_points[-1]
        assert y[-1] == 1
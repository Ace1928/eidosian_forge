import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('start, end', [(np.array([1, 0]), np.array([np.sqrt(2) / 2.0, np.sqrt(2) / 2.0])), (np.array([1, 0]), np.array([-np.sqrt(2) / 2.0, np.sqrt(2) / 2.0]))])
@pytest.mark.parametrize('t_func', [np.linspace, np.logspace])
def test_order_handling(self, start, end, t_func):
    num_t_vals = 20
    np.random.seed(789)
    forward_t_vals = t_func(0, 10, num_t_vals)
    forward_t_vals /= forward_t_vals.max()
    reverse_t_vals = np.flipud(forward_t_vals)
    shuffled_indices = np.arange(num_t_vals)
    np.random.shuffle(shuffled_indices)
    scramble_t_vals = forward_t_vals.copy()[shuffled_indices]
    forward_results = geometric_slerp(start=start, end=end, t=forward_t_vals)
    reverse_results = geometric_slerp(start=start, end=end, t=reverse_t_vals)
    scrambled_results = geometric_slerp(start=start, end=end, t=scramble_t_vals)
    assert_allclose(forward_results, np.flipud(reverse_results))
    assert_allclose(forward_results[shuffled_indices], scrambled_results)
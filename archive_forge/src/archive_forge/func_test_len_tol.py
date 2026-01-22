from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('len_tol', [0.001, 0.0001])
@pytest.mark.parametrize('locally_biased', [True, False])
def test_len_tol(self, len_tol, locally_biased):
    bounds = 4 * [(-10.0, 10.0)]
    res = direct(self.sphere, bounds=bounds, len_tol=len_tol, vol_tol=1e-30, locally_biased=locally_biased)
    assert res.status == 5
    assert res.success
    assert_allclose(res.x, np.zeros((4,)))
    message = f'The side length measure of the hyperrectangle containing the lowest function value found is below len_tol={len_tol}'
    assert res.message == message
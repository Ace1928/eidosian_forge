from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('locally_biased', [True, False])
def test_f_circle_with_args(self, locally_biased):
    bounds = 2 * [(-2.0, 2.0)]
    res = direct(self.circle_with_args, bounds, args=(1, 1), maxfun=1250, locally_biased=locally_biased)
    assert_allclose(res.x, np.array([1.0, 1.0]), rtol=1e-05)
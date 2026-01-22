from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('locally_biased', [True, False])
def test_inf_fun(self, locally_biased):
    bounds = [(-5.0, 5.0)] * 2
    result = direct(self.inf_fun, bounds, locally_biased=locally_biased)
    assert result is not None
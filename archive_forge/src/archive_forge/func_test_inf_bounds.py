from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
def test_inf_bounds(self):
    error_msg = 'Bounds must not be inf.'
    bounds = Bounds([-np.inf, -1], [-2, np.inf])
    with pytest.raises(ValueError, match=error_msg):
        direct(self.styblinski_tang, bounds)
from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('bounds', [Bounds([-1.0, -1], [-2, 1]), Bounds([-np.nan, -1], [-2, np.nan])])
def test_incorrect_bounds(self, bounds):
    error_msg = 'Bounds are not consistent min < max'
    with pytest.raises(ValueError, match=error_msg):
        direct(self.styblinski_tang, bounds)
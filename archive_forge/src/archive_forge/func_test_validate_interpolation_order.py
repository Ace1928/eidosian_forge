import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@pytest.mark.parametrize('dtype', [bool, int, np.uint8, np.uint16, float, np.float32, np.float64])
@pytest.mark.parametrize('order', [None, -1, 0, 1, 2, 3, 4, 5, 6])
def test_validate_interpolation_order(dtype, order):
    if order is None:
        assert _validate_interpolation_order(dtype, None) == 0 if dtype == bool else 1
    elif order < 0 or order > 5:
        with testing.raises(ValueError):
            _validate_interpolation_order(dtype, order)
    elif dtype == bool and order != 0:
        with pytest.raises(ValueError):
            _validate_interpolation_order(bool, order)
    else:
        assert _validate_interpolation_order(dtype, order) == order
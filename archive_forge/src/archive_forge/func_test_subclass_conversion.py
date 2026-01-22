import numpy as np
import itertools
from skimage import (
from skimage.util.dtype import _convert
from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize
def test_subclass_conversion():
    """Check subclass conversion behavior"""
    x = np.array([-1, 1])
    for dtype in float_dtype_list:
        x = x.astype(dtype)
        y = _convert(x, np.floating)
        assert y.dtype == x.dtype
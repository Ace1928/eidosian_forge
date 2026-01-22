import numpy as np
import itertools
from skimage import (
from skimage.util.dtype import _convert
from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize
def test_float_out_of_range():
    too_high = np.array([2], dtype=np.float32)
    with testing.raises(ValueError):
        img_as_int(too_high)
    too_low = np.array([-2], dtype=np.float32)
    with testing.raises(ValueError):
        img_as_int(too_low)
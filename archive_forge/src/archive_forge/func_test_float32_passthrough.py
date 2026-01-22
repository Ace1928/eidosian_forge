import numpy as np
import itertools
from skimage import (
from skimage.util.dtype import _convert
from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize
def test_float32_passthrough():
    x = np.array([-1, 1], dtype=np.float32)
    y = img_as_float(x)
    assert_equal(y.dtype, x.dtype)
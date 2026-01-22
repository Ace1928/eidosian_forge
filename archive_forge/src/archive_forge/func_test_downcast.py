import numpy as np
import itertools
from skimage import (
from skimage.util.dtype import _convert
from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize
def test_downcast():
    x = np.arange(10).astype(np.uint64)
    with expected_warnings(['Downcasting']):
        y = img_as_int(x)
    assert np.allclose(y, x.astype(np.int16))
    assert y.dtype == np.int16, y.dtype
import numpy as np
from skimage.restoration import unwrap_phase
import sys
from skimage._shared import testing
from skimage._shared.testing import (
from skimage._shared._warnings import expected_warnings
def test_unwrap_3d_all_masked():
    image = np.ma.zeros((10, 10, 10))
    image[:] = np.ma.masked
    unwrap = unwrap_phase(image)
    assert_(np.ma.isMaskedArray(unwrap))
    assert_(np.all(unwrap.mask))
    image = np.ma.zeros((10, 10, 10))
    image[:] = np.ma.masked
    image[0, 0, 0] = 0
    unwrap = unwrap_phase(image)
    assert_(np.ma.isMaskedArray(unwrap))
    assert_(np.sum(unwrap.mask) == 999)
    assert_(unwrap[0, 0, 0] == 0)
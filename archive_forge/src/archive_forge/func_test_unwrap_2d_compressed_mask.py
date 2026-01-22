import numpy as np
from skimage.restoration import unwrap_phase
import sys
from skimage._shared import testing
from skimage._shared.testing import (
from skimage._shared._warnings import expected_warnings
def test_unwrap_2d_compressed_mask():
    image = np.ma.zeros((10, 10))
    unwrap = unwrap_phase(image)
    assert_(np.all(unwrap == 0))
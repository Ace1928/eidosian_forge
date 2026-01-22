import numpy as np
from skimage.restoration import unwrap_phase
import sys
from skimage._shared import testing
from skimage._shared.testing import (
from skimage._shared._warnings import expected_warnings
@testing.parametrize('check_with_mask', (False, True))
def test_unwrap_3d(check_with_mask):
    mask = None
    x, y, z = np.ogrid[:8, :12, :16]
    image = 2 * np.pi * (x * 0.2 + y * 0.1 + z * 0.05)
    if check_with_mask:
        mask = np.zeros(image.shape, dtype=bool)
        mask[4:6, 4:6, 1:3] = True
    check_unwrap(image, mask)
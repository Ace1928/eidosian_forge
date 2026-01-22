import numpy as np
from skimage.restoration import unwrap_phase
import sys
from skimage._shared import testing
from skimage._shared.testing import (
from skimage._shared._warnings import expected_warnings
def check_unwrap(image, mask=None):
    image_wrapped = np.angle(np.exp(1j * image))
    if mask is not None:
        print('Testing a masked image')
        image = np.ma.array(image, mask=mask, fill_value=0.5)
        image_wrapped = np.ma.array(image_wrapped, mask=mask, fill_value=0.5)
    image_unwrapped = unwrap_phase(image_wrapped, rng=0)
    assert_phase_almost_equal(image_unwrapped, image)
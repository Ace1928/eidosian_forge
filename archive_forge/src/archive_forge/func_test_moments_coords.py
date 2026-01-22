import itertools
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import draw
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.measure import (
def test_moments_coords():
    image = np.zeros((20, 20), dtype=np.float64)
    image[13:17, 13:17] = 1
    mu_image = moments(image)
    coords = np.array([[r, c] for r in range(13, 17) for c in range(13, 17)], dtype=np.float64)
    mu_coords = moments_coords(coords)
    assert_almost_equal(mu_coords, mu_image)
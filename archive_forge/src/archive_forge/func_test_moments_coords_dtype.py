import itertools
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import draw
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.measure import (
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_moments_coords_dtype(dtype):
    image = np.zeros((20, 20), dtype=dtype)
    image[13:17, 13:17] = 1
    expected_dtype = _supported_float_type(dtype)
    mu_image = moments(image)
    assert mu_image.dtype == expected_dtype
    coords = np.array([[r, c] for r in range(13, 17) for c in range(13, 17)], dtype=dtype)
    mu_coords = moments_coords(coords)
    assert mu_coords.dtype == expected_dtype
    assert_almost_equal(mu_coords, mu_image)
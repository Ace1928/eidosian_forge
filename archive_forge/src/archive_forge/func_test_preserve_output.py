import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage._shared.testing import assert_stacklevel
from skimage.filters import difference_of_gaussians, gaussian
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_preserve_output(dtype):
    image = np.arange(9, dtype=dtype).reshape((3, 3))
    out = np.zeros_like(image, dtype=dtype)
    gaussian_image = gaussian(image, sigma=1, out=out, preserve_range=True)
    assert gaussian_image is out
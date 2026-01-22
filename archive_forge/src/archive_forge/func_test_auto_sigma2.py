import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage._shared.testing import assert_stacklevel
from skimage.filters import difference_of_gaussians, gaussian
@pytest.mark.parametrize('s', [1, (1, 2)])
def test_auto_sigma2(s):
    image = np.random.rand(10, 10)
    im1 = gaussian(image, sigma=s, preserve_range=True)
    s2 = 1.6 * np.array(s)
    im2 = gaussian(image, sigma=s2, preserve_range=True)
    dog = im1 - im2
    dog2 = difference_of_gaussians(image, s, s2)
    assert np.allclose(dog, dog2)
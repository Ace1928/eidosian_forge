import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage._shared.testing import assert_stacklevel
from skimage.filters import difference_of_gaussians, gaussian
def test_preserve_range():
    """Test preserve_range parameter."""
    ones = np.ones((2, 2), dtype=np.int64)
    filtered_ones = gaussian(ones, preserve_range=False)
    assert np.all(filtered_ones == filtered_ones[0, 0])
    assert filtered_ones[0, 0] < 1e-10
    filtered_preserved = gaussian(ones, preserve_range=True)
    assert np.all(filtered_preserved == 1.0)
    img = np.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, sigma=1)
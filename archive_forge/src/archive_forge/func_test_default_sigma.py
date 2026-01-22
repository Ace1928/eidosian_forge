import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage._shared.testing import assert_stacklevel
from skimage.filters import difference_of_gaussians, gaussian
def test_default_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.0
    assert_array_equal(gaussian(a, preserve_range=True), gaussian(a, preserve_range=True, sigma=1))
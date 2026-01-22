import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage._shared.testing import assert_stacklevel
from skimage.filters import difference_of_gaussians, gaussian
def test_null_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.0
    assert np.all(gaussian(a, sigma=0, preserve_range=True) == a)
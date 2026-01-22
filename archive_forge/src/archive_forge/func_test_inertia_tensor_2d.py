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
def test_inertia_tensor_2d(dtype):
    image = np.zeros((40, 40), dtype=dtype)
    image[15:25, 5:35] = 1
    expected_dtype = _supported_float_type(image.dtype)
    T = inertia_tensor(image)
    assert T.dtype == expected_dtype
    assert T[0, 0] > T[1, 1]
    np.testing.assert_allclose(T[0, 1], 0)
    v0, v1 = inertia_tensor_eigvals(image, T=T)
    assert v0.dtype == expected_dtype
    assert v1.dtype == expected_dtype
    np.testing.assert_allclose(np.sqrt(v0 / v1), 3, rtol=0.01, atol=0.05)
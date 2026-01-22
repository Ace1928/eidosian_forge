import itertools
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import draw
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.measure import (
@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.float32, np.float64])
@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize('ndim', [2, 3, 4])
def test_analytical_moments_calculation(dtype, order, ndim):
    if ndim == 2:
        shape = (256, 256)
    elif ndim == 3:
        shape = (64, 64, 64)
    else:
        shape = (16,) * ndim
    rng = np.random.default_rng(1234)
    if np.dtype(dtype).kind in 'iu':
        x = rng.integers(0, 256, shape, dtype=dtype)
    else:
        x = rng.standard_normal(shape, dtype=dtype)
    m1 = moments_central(x, center=None, order=order)
    m2 = moments_central(x, center=centroid(x), order=order)
    thresh = 0.00015 if x.dtype == np.float32 else 1e-09
    compare_moments(m1, m2, thresh=thresh)
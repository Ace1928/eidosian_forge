import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less, assert_equal
from skimage import img_as_float
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.data import camera, retina
from skimage.filters import frangi, hessian, meijering, sato
from skimage.util import crop, invert
@pytest.mark.parametrize('func', [meijering, sato, frangi, hessian])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_ridge_output_dtype(func, dtype):
    img = img_as_float(camera()).astype(dtype, copy=False)
    assert func(img).dtype == _supported_float_type(img.dtype)
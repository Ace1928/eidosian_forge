import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import chan_vese
@pytest.mark.parametrize('dtype', [np.uint8, np.float16, np.float32, np.float64])
def test_chan_vese_extended_output(dtype):
    img = np.zeros((10, 10), dtype=dtype)
    img[3:6, 3:6] = 1
    result = chan_vese(img, mu=0.0, tol=1e-08, extended_output=True)
    float_dtype = _supported_float_type(dtype)
    assert result[1].dtype == float_dtype
    assert all((arr.dtype == float_dtype for arr in result[2]))
    assert_array_equal(len(result), 3)
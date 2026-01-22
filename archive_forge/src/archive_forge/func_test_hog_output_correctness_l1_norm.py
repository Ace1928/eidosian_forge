import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_hog_output_correctness_l1_norm(dtype):
    img = color.rgb2gray(data.astronaut()).astype(dtype=dtype, copy=False)
    correct_output = np.load(fetch('data/astronaut_GRAY_hog_L1.npy'))
    output = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L1', feature_vector=True, transform_sqrt=False, visualize=False)
    float_dtype = _supported_float_type(dtype)
    assert output.dtype == float_dtype
    decimal = 7 if float_dtype == np.float64 else 5
    assert_almost_equal(output, correct_output, decimal=decimal)
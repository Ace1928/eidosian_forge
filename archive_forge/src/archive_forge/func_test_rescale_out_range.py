import warnings
import numpy as np
import pytest
from numpy.testing import (
from packaging.version import Version
from skimage import data
from skimage import exposure
from skimage import util
from skimage.color import rgb2gray
from skimage.exposure.exposure import intensity_range
from skimage.util.dtype import dtype_range
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
@pytest.mark.parametrize('dtype', [np.int8, np.int32, np.float16, np.float32, np.float64])
def test_rescale_out_range(dtype):
    """Check that output range is correct.

    .. versionchanged:: 0.17
        This function used to return dtype matching the input dtype. It now
        matches the output.

    .. versionchanged:: 0.19
        float16 and float32 inputs now result in float32 output. Formerly they
        would give float64 outputs.
    """
    image = np.array([-10, 0, 10], dtype=dtype)
    out = exposure.rescale_intensity(image, out_range=(0, 127))
    assert out.dtype == _supported_float_type(image.dtype)
    assert_array_almost_equal(out, [0, 63.5, 127])
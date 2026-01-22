import numpy as np
from skimage import dtype_limits
from skimage.util.dtype import dtype_range
from skimage.util import invert
from skimage._shared.testing import assert_array_equal
def test_invert_roundtrip():
    for t, limits in dtype_range.items():
        image = np.array(limits, dtype=t)
        expected = invert(invert(image))
        assert_array_equal(image, expected)
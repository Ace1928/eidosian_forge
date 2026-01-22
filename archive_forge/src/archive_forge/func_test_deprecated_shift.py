import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
@pytest.mark.parametrize('func', [gray.erosion, gray.dilation])
@pytest.mark.parametrize('name', ['shift_x', 'shift_y'])
@pytest.mark.parametrize('value', [True, False, None])
def test_deprecated_shift(func, name, value):
    img = np.ones(10)
    func(img)
    regex = '`shift_x` and `shift_y` are deprecated'
    with pytest.warns(FutureWarning, match=regex) as record:
        func(img, **{name: value})
    assert_stacklevel(record)
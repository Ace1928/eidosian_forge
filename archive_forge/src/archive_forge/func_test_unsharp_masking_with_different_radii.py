import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.filters import unsharp_mask
@pytest.mark.parametrize('shape,multichannel', [((32, 32), False), ((15, 15, 2), True), ((17, 19, 3), True)])
@pytest.mark.parametrize('radius', [(0.0, 0.0), (1.0, 1.0), (2.0, 1.5)])
@pytest.mark.parametrize('preserve', [False, True])
def test_unsharp_masking_with_different_radii(radius, shape, multichannel, preserve):
    amount = 1.0
    dtype = np.float64
    array = (np.random.random(shape) * 96).astype(dtype)
    if preserve is False:
        array /= max(np.abs(array).max(), 1.0)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(array, radius, amount, preserve_range=preserve, channel_axis=channel_axis)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape
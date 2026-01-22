from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_array_equal
import numpy as np
from skimage.util import montage
@testing.parametrize('channel_axis', (4, -5))
def test_montage_invalid_channel_axes(channel_axis):
    arr_in = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    with testing.raises(AxisError):
        montage(arr_in, channel_axis=channel_axis)
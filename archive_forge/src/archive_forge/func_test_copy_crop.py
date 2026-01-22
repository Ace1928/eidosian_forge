import numpy as np
from skimage.util import crop
from skimage._shared.testing import assert_array_equal, assert_equal
def test_copy_crop():
    arr = np.arange(45).reshape(9, 5)
    out0 = crop(arr, 1, copy=True)
    assert out0.flags.c_contiguous
    out0[0, 0] = 100
    assert not np.any(arr == 100)
    assert not np.may_share_memory(arr, out0)
    out1 = crop(arr, 1)
    out1[0, 0] = 100
    assert arr[1, 1] == 100
    assert np.may_share_memory(arr, out1)
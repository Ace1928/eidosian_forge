import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest
@pytest.mark.parametrize('dtype', (np.byte, np.short, np.intc, int, np.longlong, np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong))
@pytest.mark.parametrize('data_already_sequential', (False, True))
def test_relabel_sequential_int_dtype_stability(data_already_sequential, dtype):
    if data_already_sequential:
        ar = np.array([1, 3, 0, 2, 5, 4], dtype=dtype)
    else:
        ar = np.array([1, 1, 5, 5, 8, 99, 42, 0], dtype=dtype)
    assert all((a.dtype == dtype for a in relabel_sequential(ar)))
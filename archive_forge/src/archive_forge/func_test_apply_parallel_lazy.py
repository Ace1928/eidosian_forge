import numpy as np
from skimage._shared.testing import assert_array_almost_equal, assert_equal
from skimage import color, data, img_as_float
from skimage.filters import threshold_local, gaussian
from skimage.util.apply_parallel import apply_parallel
import pytest
def test_apply_parallel_lazy():
    a = np.arange(144).reshape(12, 12).astype(float)
    d = da.from_array(a, chunks=(6, 6))
    expected1 = threshold_local(a, 3)
    result1 = apply_parallel(threshold_local, a, chunks=(6, 6), depth=5, extra_arguments=(3,), extra_keywords={'mode': 'reflect'}, compute=False)
    result2 = apply_parallel(threshold_local, d, depth=5, extra_arguments=(3,), extra_keywords={'mode': 'reflect'})
    assert isinstance(result1, da.Array)
    assert_array_almost_equal(result1.compute(), expected1)
    assert isinstance(result2, da.Array)
    assert_array_almost_equal(result2.compute(), expected1)
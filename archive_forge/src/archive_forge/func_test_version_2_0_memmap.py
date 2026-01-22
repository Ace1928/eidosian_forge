import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
@pytest.mark.skipif(IS_WASM, reason="memmap doesn't work correctly")
def test_version_2_0_memmap(tmpdir):
    dt = [('%d' % i * 100, float) for i in range(500)]
    d = np.ones(1000, dtype=dt)
    tf1 = os.path.join(tmpdir, f'version2_01.npy')
    tf2 = os.path.join(tmpdir, f'version2_02.npy')
    assert_raises(ValueError, format.open_memmap, tf1, mode='w+', dtype=d.dtype, shape=d.shape, version=(1, 0))
    ma = format.open_memmap(tf1, mode='w+', dtype=d.dtype, shape=d.shape, version=(2, 0))
    ma[...] = d
    ma.flush()
    ma = format.open_memmap(tf1, mode='r', max_header_size=200000)
    assert_array_equal(ma, d)
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', UserWarning)
        ma = format.open_memmap(tf2, mode='w+', dtype=d.dtype, shape=d.shape, version=None)
        assert_(w[0].category is UserWarning)
        ma[...] = d
        ma.flush()
    ma = format.open_memmap(tf2, mode='r', max_header_size=200000)
    assert_array_equal(ma, d)
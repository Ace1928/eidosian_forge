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
@pytest.mark.slow
def test_memmap_roundtrip(tmpdir):
    for i, arr in enumerate(basic_arrays + record_arrays):
        if arr.dtype.hasobject:
            continue
        nfn = os.path.join(tmpdir, f'normal{i}.npy')
        mfn = os.path.join(tmpdir, f'memmap{i}.npy')
        with open(nfn, 'wb') as fp:
            format.write_array(fp, arr)
        fortran_order = arr.flags.f_contiguous and (not arr.flags.c_contiguous)
        ma = format.open_memmap(mfn, mode='w+', dtype=arr.dtype, shape=arr.shape, fortran_order=fortran_order)
        ma[...] = arr
        ma.flush()
        with open(nfn, 'rb') as fp:
            normal_bytes = fp.read()
        with open(mfn, 'rb') as fp:
            memmap_bytes = fp.read()
        assert_equal_(normal_bytes, memmap_bytes)
        ma = format.open_memmap(nfn, mode='r')
        ma.flush()
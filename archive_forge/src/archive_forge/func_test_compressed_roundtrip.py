import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_compressed_roundtrip(tmpdir):
    arr = np.random.rand(200, 200)
    npz_file = os.path.join(tmpdir, 'compressed.npz')
    np.savez_compressed(npz_file, arr=arr)
    with np.load(npz_file) as npz:
        arr1 = npz['arr']
    assert_array_equal(arr, arr1)
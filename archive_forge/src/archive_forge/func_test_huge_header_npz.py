import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_huge_header_npz(tmpdir):
    f = os.path.join(tmpdir, f'large_header.npz')
    arr = np.array(1, dtype='i,' * 10000 + 'i')
    with pytest.warns(UserWarning, match='.*format 2.0'):
        np.savez(f, arr=arr)
    with pytest.raises(ValueError, match='Header.*large'):
        np.load(f)['arr']
    with pytest.raises(ValueError, match='Header.*large'):
        np.load(f, max_header_size=20000)['arr']
    res = np.load(f, allow_pickle=True)['arr']
    assert_array_equal(res, arr)
    res = np.load(f, max_header_size=180000)['arr']
    assert_array_equal(res, arr)
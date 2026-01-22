import os
from platform import machine
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import (
from ..testing import suppress_warnings
def test_int_abs():
    for itype in sctypes['int']:
        info = np.iinfo(itype)
        in_arr = np.array([info.min, info.max], dtype=itype)
        idtype = np.dtype(itype)
        udtype = np.dtype(idtype.str.replace('i', 'u'))
        assert udtype.kind == 'u'
        assert idtype.itemsize == udtype.itemsize
        mn, mx = in_arr
        e_mn = int(mx) + 1
        assert int_abs(mx) == mx
        assert int_abs(mn) == e_mn
        assert_array_equal(int_abs(in_arr), [e_mn, mx])
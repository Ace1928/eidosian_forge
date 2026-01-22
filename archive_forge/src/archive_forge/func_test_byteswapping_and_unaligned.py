import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize(['dtype', 'value'], [('i2', 1), ('u2', 1), ('i4', 66051), ('u4', 66051), ('i8', 283686952306183), ('u8', 283686952306183), ('float16', 3.07e-05), ('float32', 9.2557e-41), ('complex64', 9.2557e-41 + 2.8622554e-29j), ('float64', -1.758571353180402e-24), ('complex128', repr(5.406409232372729e-29 - 1.758571353180402e-24j)), ('longdouble', 283686952306183), ('clongdouble', repr(283686952306183 + 19873150604823 * 1j)), ('U2', '\U00010203\U000a0b0c')])
@pytest.mark.parametrize('swap', [True, False])
def test_byteswapping_and_unaligned(dtype, value, swap):
    dtype = np.dtype(dtype)
    data = [f'x,{value}\n']
    if swap:
        dtype = dtype.newbyteorder()
    full_dt = np.dtype([('a', 'S1'), ('b', dtype)], align=False)
    assert full_dt.fields['b'][1] == 1
    res = np.loadtxt(data, dtype=full_dt, delimiter=',', encoding=None, max_rows=1)
    assert res['b'] == dtype.type(value)
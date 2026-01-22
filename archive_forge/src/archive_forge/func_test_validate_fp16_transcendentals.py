import numpy as np
import os
from os import path
import sys
import pytest
from ctypes import c_longlong, c_double, c_float, c_int, cast, pointer, POINTER
from numpy.testing import assert_array_max_ulp
from numpy.testing._private.utils import _glibc_older_than
from numpy.core._multiarray_umath import __cpu_features__
@pytest.mark.parametrize('ufunc', UNARY_OBJECT_UFUNCS)
def test_validate_fp16_transcendentals(self, ufunc):
    with np.errstate(all='ignore'):
        arr = np.arange(65536, dtype=np.int16)
        datafp16 = np.frombuffer(arr.tobytes(), dtype=np.float16)
        datafp32 = datafp16.astype(np.float32)
        assert_array_max_ulp(ufunc(datafp16), ufunc(datafp32), maxulp=1, dtype=np.float16)
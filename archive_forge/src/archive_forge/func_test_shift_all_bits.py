import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
@pytest.mark.parametrize('type_code', np.typecodes['AllInteger'])
@pytest.mark.parametrize('op', [operator.rshift, operator.lshift], ids=['>>', '<<'])
def test_shift_all_bits(self, type_code, op):
    """Shifts where the shift amount is the width of the type or wider """
    if USING_CLANG_CL and type_code in ('l', 'L') and (op is operator.lshift):
        pytest.xfail('Failing on clang-cl builds')
    dt = np.dtype(type_code)
    nbits = dt.itemsize * 8
    for val in [5, -5]:
        for shift in [nbits, nbits + 4]:
            val_scl = np.array(val).astype(dt)[()]
            shift_scl = dt.type(shift)
            res_scl = op(val_scl, shift_scl)
            if val_scl < 0 and op is operator.rshift:
                assert_equal(res_scl, -1)
            else:
                assert_equal(res_scl, 0)
            val_arr = np.array([val_scl] * 32, dtype=dt)
            shift_arr = np.array([shift] * 32, dtype=dt)
            res_arr = op(val_arr, shift_arr)
            assert_equal(res_arr, res_scl)
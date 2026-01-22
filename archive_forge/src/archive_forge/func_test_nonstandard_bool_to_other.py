import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'] + np.typecodes['AllFloat'])
def test_nonstandard_bool_to_other(self, dtype):
    nonstandard_bools = np.array([0, 3, -7], dtype=np.int8).view(bool)
    res = nonstandard_bools.astype(dtype)
    expected = [0, 1, 1]
    assert_array_equal(res, expected)
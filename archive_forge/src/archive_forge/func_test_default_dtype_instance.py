from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('dtype_char', np.typecodes['All'])
def test_default_dtype_instance(self, dtype_char):
    if dtype_char in 'SU':
        dtype = np.dtype(dtype_char + '1')
    elif dtype_char == 'V':
        dtype = np.dtype('V8')
    else:
        dtype = np.dtype(dtype_char)
    discovered_dtype, _ = _discover_array_parameters([], type(dtype))
    assert discovered_dtype == dtype
    assert discovered_dtype.itemsize == dtype.itemsize
import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'])
@pytest.mark.filterwarnings('error:.*integer via a float.*:DeprecationWarning')
def test_integer_signs(dtype):
    dtype = np.dtype(dtype)
    assert np.loadtxt(['+2'], dtype=dtype) == 2
    if dtype.kind == 'u':
        with pytest.raises(ValueError):
            np.loadtxt(['-1\n'], dtype=dtype)
    else:
        assert np.loadtxt(['-2\n'], dtype=dtype) == -2
    for sign in ['++', '+-', '--', '-+']:
        with pytest.raises(ValueError):
            np.loadtxt([f'{sign}2\n'], dtype=dtype)
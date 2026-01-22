import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'] + 'efgdFDG' + '?')
def test_no_thousands_support(dtype):
    if dtype == 'e':
        pytest.skip('half assignment currently uses Python float converter')
    if dtype in 'eG':
        pytest.xfail('clongdouble assignment is buggy (uses `complex`?).')
    assert int('1_1') == float('1_1') == complex('1_1') == 11
    with pytest.raises(ValueError):
        np.loadtxt(['1_1\n'], dtype=dtype)
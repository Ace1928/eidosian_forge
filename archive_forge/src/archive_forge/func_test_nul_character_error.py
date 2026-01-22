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
def test_nul_character_error(dtype):
    if dtype.lower() == 'g':
        pytest.xfail('longdouble/clongdouble assignment may misbehave.')
    with pytest.raises(ValueError):
        np.loadtxt(['1\x00'], dtype=dtype, delimiter=',', quotechar='"')
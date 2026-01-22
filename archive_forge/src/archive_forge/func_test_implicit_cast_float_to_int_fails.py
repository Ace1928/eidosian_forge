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
def test_implicit_cast_float_to_int_fails(dtype):
    txt = StringIO('1.0, 2.1, 3.7\n4, 5, 6')
    with pytest.raises(ValueError):
        np.loadtxt(txt, dtype=dtype, delimiter=',')
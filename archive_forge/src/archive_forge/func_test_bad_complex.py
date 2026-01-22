import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
@pytest.mark.parametrize('dtype', 'FD')
@pytest.mark.parametrize('field', ['1 +2j', '1+ 2j', '1+2 j', '1+-+3', '(1j', '(1', '(1+2j', '1+2j)'])
def test_bad_complex(dtype, field):
    with pytest.raises(ValueError):
        np.loadtxt([field + '\n'], dtype=dtype, delimiter=',')
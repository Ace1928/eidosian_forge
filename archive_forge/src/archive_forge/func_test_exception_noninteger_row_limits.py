import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('param', ('skiprows', 'max_rows'))
def test_exception_noninteger_row_limits(param):
    with pytest.raises(TypeError, match='argument must be an integer'):
        np.loadtxt('foo.bar', **{param: 1.0})
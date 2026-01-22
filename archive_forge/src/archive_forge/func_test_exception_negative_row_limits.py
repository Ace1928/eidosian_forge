import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('param', ('skiprows', 'max_rows'))
def test_exception_negative_row_limits(param):
    """skiprows and max_rows should raise for negative parameters."""
    with pytest.raises(ValueError, match='argument must be nonnegative'):
        np.loadtxt('foo.bar', **{param: -3})
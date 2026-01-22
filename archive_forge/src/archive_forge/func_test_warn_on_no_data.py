import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('data', ('', '\n\n\n', '# 1 2 3\n# 4 5 6\n'))
@pytest.mark.parametrize('ndmin', (0, 1, 2))
@pytest.mark.parametrize('usecols', [None, (1, 2, 3)])
def test_warn_on_no_data(data, ndmin, usecols):
    """Check that a UserWarning is emitted when no data is read from input."""
    if usecols is not None:
        expected_shape = (0, 3)
    elif ndmin == 2:
        expected_shape = (0, 1)
    else:
        expected_shape = (0,)
    txt = StringIO(data)
    with pytest.warns(UserWarning, match='input contained no data'):
        res = np.loadtxt(txt, ndmin=ndmin, usecols=usecols)
    assert res.shape == expected_shape
    with NamedTemporaryFile(mode='w') as fh:
        fh.write(data)
        fh.seek(0)
        with pytest.warns(UserWarning, match='input contained no data'):
            res = np.loadtxt(txt, ndmin=ndmin, usecols=usecols)
        assert res.shape == expected_shape
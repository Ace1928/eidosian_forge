import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_ragged_usecols():
    txt = StringIO('0,0,XXX\n0,XXX,0,XXX\n0,XXX,XXX,0,XXX\n')
    expected = np.array([[0, 0], [0, 0], [0, 0]])
    res = np.loadtxt(txt, dtype=float, delimiter=',', usecols=[0, -2])
    assert_equal(res, expected)
    txt = StringIO('0,0,XXX\n0\n0,XXX,XXX,0,XXX\n')
    with pytest.raises(ValueError, match='invalid column index -2 at row 2 with 1 columns'):
        np.loadtxt(txt, dtype=float, delimiter=',', usecols=[0, -2])
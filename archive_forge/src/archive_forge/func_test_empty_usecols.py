import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_empty_usecols():
    txt = StringIO('0,0,XXX\n0,XXX,0,XXX\n0,XXX,XXX,0,XXX\n')
    res = np.loadtxt(txt, dtype=np.dtype([]), delimiter=',', usecols=[])
    assert res.shape == (3,)
    assert res.dtype == np.dtype([])
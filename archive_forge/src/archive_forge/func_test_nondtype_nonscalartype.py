import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_nondtype_nonscalartype(self):
    assert not np.issubdtype(np.float32, 'float64')
    assert not np.issubdtype(np.float32, 'f8')
    assert not np.issubdtype(np.int32, str)
    assert not np.issubdtype(np.int32, 'int64')
    assert not np.issubdtype(np.str_, 'void')
    assert not np.issubdtype(np.int8, int)
    assert not np.issubdtype(np.float32, float)
    assert not np.issubdtype(np.complex64, complex)
    assert not np.issubdtype(np.float32, 'float')
    assert not np.issubdtype(np.float64, 'f')
    assert np.issubdtype(np.float64, 'float64')
    assert np.issubdtype(np.float64, 'f8')
    assert np.issubdtype(np.str_, str)
    assert np.issubdtype(np.int64, 'int64')
    assert np.issubdtype(np.void, 'void')
    assert np.issubdtype(np.int8, np.integer)
    assert np.issubdtype(np.float32, np.floating)
    assert np.issubdtype(np.complex64, np.complexfloating)
    assert np.issubdtype(np.float64, 'float')
    assert np.issubdtype(np.float32, 'f')
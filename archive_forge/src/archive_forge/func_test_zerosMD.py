import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_zerosMD(self):
    """Check creation of multi-dimensional objects"""
    h = np.zeros((2, 3), dtype=self._descr)
    assert_(normalize_descr(self._descr) == h.dtype.descr)
    assert_(h.dtype['z'].name == 'uint8')
    assert_(h.dtype['z'].char == 'B')
    assert_(h.dtype['z'].type == np.uint8)
    assert_equal(h['z'], np.zeros((2, 3), dtype='u1'))
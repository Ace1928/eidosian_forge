import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_nested1_acessors(self):
    """Check reading the nested fields of a nested array (1st level)"""
    h = np.array(self._buffer, dtype=self._descr)
    if not self.multiple_rows:
        assert_equal(h['Info']['value'], np.array(self._buffer[1][0], dtype='c16'))
        assert_equal(h['Info']['y2'], np.array(self._buffer[1][1], dtype='f8'))
        assert_equal(h['info']['Name'], np.array(self._buffer[3][0], dtype='U2'))
        assert_equal(h['info']['Value'], np.array(self._buffer[3][1], dtype='c16'))
    else:
        assert_equal(h['Info']['value'], np.array([self._buffer[0][1][0], self._buffer[1][1][0]], dtype='c16'))
        assert_equal(h['Info']['y2'], np.array([self._buffer[0][1][1], self._buffer[1][1][1]], dtype='f8'))
        assert_equal(h['info']['Name'], np.array([self._buffer[0][3][0], self._buffer[1][3][0]], dtype='U2'))
        assert_equal(h['info']['Value'], np.array([self._buffer[0][3][1], self._buffer[1][3][1]], dtype='c16'))
import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_integers_broadcast(self, dtype):
    if dtype == np.bool_:
        upper = 2
        lower = 0
    else:
        info = np.iinfo(dtype)
        upper = int(info.max) + 1
        lower = info.min
    self._reset_state()
    a = self.rg.integers(lower, [upper] * 10, dtype=dtype)
    self._reset_state()
    b = self.rg.integers([lower] * 10, upper, dtype=dtype)
    assert_equal(a, b)
    self._reset_state()
    c = self.rg.integers(lower, upper, size=10, dtype=dtype)
    assert_equal(a, c)
    self._reset_state()
    d = self.rg.integers(np.array([lower] * 10), np.array([upper], dtype=object), size=10, dtype=dtype)
    assert_equal(a, d)
    self._reset_state()
    e = self.rg.integers(np.array([lower] * 10), np.array([upper] * 10), size=10, dtype=dtype)
    assert_equal(a, e)
    self._reset_state()
    a = self.rg.integers(0, upper, size=10, dtype=dtype)
    self._reset_state()
    b = self.rg.integers([upper] * 10, dtype=dtype)
    assert_equal(a, b)
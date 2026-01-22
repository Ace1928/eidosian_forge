import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_flatten_dtype(self):
    """Testing flatten_dtype"""
    dt = np.dtype([('a', 'f8'), ('b', 'f8')])
    dt_flat = flatten_dtype(dt)
    assert_equal(dt_flat, [float, float])
    dt = np.dtype([('a', [('aa', '|S1'), ('ab', '|S2')]), ('b', int)])
    dt_flat = flatten_dtype(dt)
    assert_equal(dt_flat, [np.dtype('|S1'), np.dtype('|S2'), int])
    dt = np.dtype([('a', (float, 2)), ('b', (int, 3))])
    dt_flat = flatten_dtype(dt)
    assert_equal(dt_flat, [float, int])
    dt_flat = flatten_dtype(dt, True)
    assert_equal(dt_flat, [float] * 2 + [int] * 3)
    dt = np.dtype([(('a', 'A'), 'f8'), (('b', 'B'), 'f8')])
    dt_flat = flatten_dtype(dt)
    assert_equal(dt_flat, [float, float])
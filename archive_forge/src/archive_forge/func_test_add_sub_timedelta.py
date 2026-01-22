import contextlib
import itertools
import re
import unittest
import warnings
import numpy as np
from numba import jit, vectorize, njit
from numba.np.numpy_support import numpy_version
from numba.core import types, config
from numba.core.errors import TypingError
from numba.tests.support import TestCase, tag, skip_parfors_unsupported
from numba.np import npdatetime_helpers, numpy_support
def test_add_sub_timedelta(self):
    """
        Test `datetime64 + timedelta64` and `datetime64 - timedelta64`.
        """
    add = self.jit(add_usecase)
    sub = self.jit(sub_usecase)

    def check(a, b, expected):
        with self.silence_numpy_warnings():
            self.assertPreciseEqual(add(a, b), expected, (a, b))
            self.assertPreciseEqual(add(b, a), expected, (a, b))
            self.assertPreciseEqual(sub(a, -b), expected, (a, b))
            self.assertPreciseEqual(a + b, expected)
    check(DT('2014'), TD(2, 'Y'), DT('2016'))
    check(DT('2014'), TD(2, 'M'), DT('2014-03'))
    check(DT('2014'), TD(3, 'W'), DT('2014-01-16', 'W'))
    check(DT('2014'), TD(4, 'D'), DT('2014-01-05'))
    check(DT('2000'), TD(365, 'D'), DT('2000-12-31'))
    check(DT('2014-02'), TD(2, 'Y'), DT('2016-02'))
    check(DT('2014-02'), TD(2, 'M'), DT('2014-04'))
    check(DT('2014-02'), TD(2, 'D'), DT('2014-02-03'))
    check(DT('2014-01-07', 'W'), TD(2, 'W'), DT('2014-01-16', 'W'))
    check(DT('2014-02-02'), TD(27, 'D'), DT('2014-03-01'))
    check(DT('2012-02-02'), TD(27, 'D'), DT('2012-02-29'))
    check(DT('2012-02-02'), TD(2, 'W'), DT('2012-02-16'))
    check(DT('2000-01-01T01:02:03Z'), TD(2, 'h'), DT('2000-01-01T03:02:03Z'))
    check(DT('2000-01-01T01:02:03Z'), TD(2, 'ms'), DT('2000-01-01T01:02:03.002Z'))
    for dt_str in ('600', '601', '604', '801', '1900', '1904', '2200', '2300', '2304', '2400', '6001'):
        for dt_suffix in ('', '-01', '-12'):
            dt = DT(dt_str + dt_suffix)
            for td in [TD(2, 'D'), TD(2, 'W'), TD(100, 'D'), TD(10000, 'D'), TD(-100, 'D'), TD(-10000, 'D'), TD(100, 'W'), TD(10000, 'W'), TD(-100, 'W'), TD(-10000, 'W'), TD(100, 'M'), TD(10000, 'M'), TD(-100, 'M'), TD(-10000, 'M')]:
                self.assertEqual(add(dt, td), dt + td, (dt, td))
                self.assertEqual(add(td, dt), dt + td, (dt, td))
                self.assertEqual(sub(dt, -td), dt + td, (dt, td))
    check(DT('NaT'), TD(2), DT('NaT'))
    check(DT('NaT', 's'), TD(2, 'h'), DT('NaT', 's'))
    check(DT('NaT', 's'), TD(2, 'ms'), DT('NaT', 'ms'))
    check(DT('2014'), TD('NaT', 'W'), DT('NaT', 'W'))
    check(DT('2014-01-01'), TD('NaT', 'W'), DT('NaT', 'D'))
    check(DT('NaT', 's'), TD('NaT', 'ms'), DT('NaT', 'ms'))
    for f in (add, sub):
        with self.assertRaises((TypeError, TypingError)):
            f(DT(1, '2014-01-01'), TD(1, 'Y'))
        with self.assertRaises((TypeError, TypingError)):
            f(DT(1, '2014-01-01'), TD(1, 'M'))
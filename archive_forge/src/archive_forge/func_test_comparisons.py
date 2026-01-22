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
def test_comparisons(self):
    eq = self.jit(eq_usecase)
    ne = self.jit(ne_usecase)
    lt = self.jit(lt_usecase)
    le = self.jit(le_usecase)
    gt = self.jit(gt_usecase)
    ge = self.jit(ge_usecase)

    def check_eq(a, b, expected):
        expected_val = expected
        not_expected_val = not expected
        if np.isnat(a) or np.isnat(b):
            expected_val = False
            not_expected_val = True
            self.assertFalse(le(a, b), (a, b))
            self.assertFalse(ge(a, b), (a, b))
            self.assertFalse(le(b, a), (a, b))
            self.assertFalse(ge(b, a), (a, b))
            self.assertFalse(lt(a, b), (a, b))
            self.assertFalse(gt(a, b), (a, b))
            self.assertFalse(lt(b, a), (a, b))
            self.assertFalse(gt(b, a), (a, b))
        with self.silence_numpy_warnings():
            self.assertPreciseEqual(eq(a, b), expected_val, (a, b, expected))
            self.assertPreciseEqual(eq(b, a), expected_val, (a, b, expected))
            self.assertPreciseEqual(ne(a, b), not_expected_val, (a, b, expected))
            self.assertPreciseEqual(ne(b, a), not_expected_val, (a, b, expected))
            if expected_val:
                self.assertTrue(le(a, b), (a, b))
                self.assertTrue(ge(a, b), (a, b))
                self.assertTrue(le(b, a), (a, b))
                self.assertTrue(ge(b, a), (a, b))
                self.assertFalse(lt(a, b), (a, b))
                self.assertFalse(gt(a, b), (a, b))
                self.assertFalse(lt(b, a), (a, b))
                self.assertFalse(gt(b, a), (a, b))
            self.assertPreciseEqual(a == b, expected_val)

    def check_lt(a, b, expected):
        expected_val = expected
        not_expected_val = not expected
        if np.isnat(a) or np.isnat(b):
            expected_val = False
            not_expected_val = False
        with self.silence_numpy_warnings():
            lt = self.jit(lt_usecase)
            self.assertPreciseEqual(lt(a, b), expected_val, (a, b, expected))
            self.assertPreciseEqual(gt(b, a), expected_val, (a, b, expected))
            self.assertPreciseEqual(ge(a, b), not_expected_val, (a, b, expected))
            self.assertPreciseEqual(le(b, a), not_expected_val, (a, b, expected))
            if expected_val:
                check_eq(a, b, False)
            self.assertPreciseEqual(a < b, expected_val)
    check_eq(DT('2014'), DT('2017'), False)
    check_eq(DT('2014'), DT('2014-01'), True)
    check_eq(DT('2014'), DT('2014-01-01'), True)
    check_eq(DT('2014'), DT('2014-01-01', 'W'), True)
    check_eq(DT('2014-01'), DT('2014-01-01', 'W'), True)
    check_eq(DT('2014-01-01'), DT('2014-01-01', 'W'), False)
    check_eq(DT('2014-01-02'), DT('2014-01-06', 'W'), True)
    check_eq(DT('2014-01-01T00:01:00Z', 's'), DT('2014-01-01T00:01Z', 'm'), True)
    check_eq(DT('2014-01-01T00:01:01Z', 's'), DT('2014-01-01T00:01Z', 'm'), False)
    check_lt(DT('NaT', 'Y'), DT('2017'), True)
    check_eq(DT('NaT'), DT('NaT'), True)
    dts = self.datetime_samples()
    for a in dts:
        a_unit = a.dtype.str.split('[')[1][:-1]
        i = all_units.index(a_unit)
        units = all_units[i:i + 6]
        for unit in units:
            b = a.astype('M8[%s]' % unit)
            if not npdatetime_helpers.same_kind(value_unit(a), value_unit(b)):
                continue
            check_eq(a, b, True)
            check_lt(a, b + np.timedelta64(1, unit), True)
            check_lt(b - np.timedelta64(1, unit), a, True)
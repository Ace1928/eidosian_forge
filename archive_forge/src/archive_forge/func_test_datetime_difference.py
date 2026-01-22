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
def test_datetime_difference(self):
    """
        Test `datetime64 - datetime64`.
        """
    sub = self.jit(sub_usecase)

    def check(a, b, expected=None):
        with self.silence_numpy_warnings():
            self.assertPreciseEqual(sub(a, b), a - b, (a, b))
            self.assertPreciseEqual(sub(b, a), b - a, (a, b))
            self.assertPreciseEqual(a - b, expected)
    check(DT('2014'), DT('2017'), TD(-3, 'Y'))
    check(DT('2014-02'), DT('2017-01'), TD(-35, 'M'))
    check(DT('2014-02-28'), DT('2015-03-01'), TD(-366, 'D'))
    check(DT('NaT', 'M'), DT('2000'), TD('NaT', 'M'))
    check(DT('NaT', 'M'), DT('2000-01-01'), TD('NaT', 'D'))
    check(DT('NaT'), DT('NaT'), TD('NaT'))
    with self.silence_numpy_warnings():
        dts = self.datetime_samples()
        for a, b in itertools.product(dts, dts):
            if not npdatetime_helpers.same_kind(value_unit(a), value_unit(b)):
                continue
            self.assertPreciseEqual(sub(a, b), a - b, (a, b))
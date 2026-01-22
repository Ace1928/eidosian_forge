import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
@pytest.mark.skipif(IS_WASM, reason='no wasm fp exception support')
def test_errobj(self):
    olderrobj = np.geterrobj()
    self.called = 0
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with np.errstate(divide='warn'):
                np.seterrobj([20000, 1, None])
                np.array([1.0]) / np.array([0.0])
                assert_equal(len(w), 1)

        def log_err(*args):
            self.called += 1
            extobj_err = args
            assert_(len(extobj_err) == 2)
            assert_('divide' in extobj_err[0])
        with np.errstate(divide='ignore'):
            np.seterrobj([20000, 3, log_err])
            np.array([1.0]) / np.array([0.0])
        assert_equal(self.called, 1)
        np.seterrobj(olderrobj)
        with np.errstate(divide='ignore'):
            np.divide(1.0, 0.0, extobj=[20000, 3, log_err])
        assert_equal(self.called, 2)
    finally:
        np.seterrobj(olderrobj)
        del self.called
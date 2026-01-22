import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def test_float_modulus_corner_cases(self):
    for dt in np.typecodes['Float']:
        b = np.array(1.0, dtype=dt)
        a = np.nextafter(np.array(0.0, dtype=dt), -b)
        rem = operator.mod(a, b)
        assert_(rem <= b, 'dt: %s' % dt)
        rem = operator.mod(-a, -b)
        assert_(rem >= -b, 'dt: %s' % dt)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in remainder')
        sup.filter(RuntimeWarning, 'divide by zero encountered in remainder')
        sup.filter(RuntimeWarning, 'divide by zero encountered in floor_divide')
        sup.filter(RuntimeWarning, 'divide by zero encountered in divmod')
        sup.filter(RuntimeWarning, 'invalid value encountered in divmod')
        for dt in np.typecodes['Float']:
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            finf = np.array(np.inf, dtype=dt)
            fnan = np.array(np.nan, dtype=dt)
            rem = operator.mod(fone, fzer)
            assert_(np.isnan(rem), 'dt: %s' % dt)
            rem = operator.mod(fone, fnan)
            assert_(np.isnan(rem), 'dt: %s' % dt)
            rem = operator.mod(finf, fone)
            assert_(np.isnan(rem), 'dt: %s' % dt)
            for op in [floordiv_and_mod, divmod]:
                div, mod = op(fone, fzer)
                assert_(np.isinf(div)) and assert_(np.isnan(mod))
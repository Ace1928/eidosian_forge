import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_float_remainder_corner_cases(self):
    for dt in np.typecodes['Float']:
        fone = np.array(1.0, dtype=dt)
        fzer = np.array(0.0, dtype=dt)
        fnan = np.array(np.nan, dtype=dt)
        b = np.array(1.0, dtype=dt)
        a = np.nextafter(np.array(0.0, dtype=dt), -b)
        rem = np.remainder(a, b)
        assert_(rem <= b, 'dt: %s' % dt)
        rem = np.remainder(-a, -b)
        assert_(rem >= -b, 'dt: %s' % dt)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in remainder')
        sup.filter(RuntimeWarning, 'invalid value encountered in fmod')
        for dt in np.typecodes['Float']:
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            finf = np.array(np.inf, dtype=dt)
            fnan = np.array(np.nan, dtype=dt)
            rem = np.remainder(fone, fzer)
            assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
            rem = np.remainder(finf, fone)
            fmod = np.fmod(finf, fone)
            assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
            assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
            rem = np.remainder(finf, finf)
            fmod = np.fmod(finf, fone)
            assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
            assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
            rem = np.remainder(finf, fzer)
            fmod = np.fmod(finf, fzer)
            assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
            assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
            rem = np.remainder(fone, fnan)
            fmod = np.fmod(fone, fnan)
            assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
            assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
            rem = np.remainder(fnan, fzer)
            fmod = np.fmod(fnan, fzer)
            assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
            assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, rem))
            rem = np.remainder(fnan, fone)
            fmod = np.fmod(fnan, fone)
            assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
            assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, rem))
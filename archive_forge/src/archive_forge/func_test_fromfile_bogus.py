import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_fromfile_bogus(self):
    with temppath() as path:
        with open(path, 'w') as f:
            f.write('1. 2. 3. flop 4.\n')
        with assert_warns(DeprecationWarning):
            res = np.fromfile(path, dtype=float, sep=' ')
    assert_equal(res, np.array([1.0, 2.0, 3.0]))
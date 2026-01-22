import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_fromstring_best_effort_float(self):
    with assert_warns(DeprecationWarning):
        assert_equal(np.fromstring('1,234', dtype=float, sep=' '), np.array([1.0]))
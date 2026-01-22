import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_fromstring_foreign_value(self):
    with assert_warns(DeprecationWarning):
        b = np.fromstring('1,234', dtype=np.longdouble, sep=' ')
        assert_array_equal(b[0], 1)
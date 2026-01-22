import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_fromstring_empty():
    with assert_warns(DeprecationWarning):
        assert_equal(np.fromstring('xxxxx', sep='x'), np.array([]))
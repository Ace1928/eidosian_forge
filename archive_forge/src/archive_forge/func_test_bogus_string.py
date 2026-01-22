import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_bogus_string():
    assert_raises(ValueError, np.longdouble, 'spam')
    assert_raises(ValueError, np.longdouble, '1.0 flub')
import sys
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, IS_MUSL
from numpy.core.tests._locales import CommaDecimalPointLocale
from io import StringIO
def test_locale_double(self):
    assert_equal(str(np.double(1.2)), str(float(1.2)))
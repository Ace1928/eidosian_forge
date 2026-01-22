import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
@pytest.mark.skip(reason='cexp(nan + 0I) is wrong on most platforms')
def test_special_values2(self):
    check = check_complex_value
    f = np.exp
    check(f, np.nan, 0, np.nan, 0)
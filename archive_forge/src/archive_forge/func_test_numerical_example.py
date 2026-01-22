from __future__ import division
from __future__ import absolute_import
import sys
import math
from uncertainties import ufloat
import uncertainties.core as uncert_core
import uncertainties.umath_core as umath_core
from . import test_uncertainties
def test_numerical_example():
    """Test specific numerical examples"""
    x = ufloat(3.14, 0.01)
    result = umath_core.sin(x)
    assert '%.6f +/- %.6f' % (result.nominal_value, result.std_dev) == '0.001593 +/- 0.010000'
    assert '%.11f' % umath_core.sin(3) == '0.14112000806'
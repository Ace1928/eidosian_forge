from scipy.constants import find, value, ConstantWarning, c, speed_of_light
from numpy.testing import (assert_equal, assert_, assert_almost_equal,
import scipy.constants._codata as _cd
def test_exact_values():
    with suppress_warnings() as sup:
        sup.filter(ConstantWarning)
        for key in _cd.exact_values:
            assert_((_cd.exact_values[key][0] - value(key)) / value(key) == 0)
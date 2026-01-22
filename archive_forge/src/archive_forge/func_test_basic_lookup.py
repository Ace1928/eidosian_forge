from scipy.constants import find, value, ConstantWarning, c, speed_of_light
from numpy.testing import (assert_equal, assert_, assert_almost_equal,
import scipy.constants._codata as _cd
def test_basic_lookup():
    assert_equal('%d %s' % (_cd.c, _cd.unit('speed of light in vacuum')), '299792458 m s^-1')
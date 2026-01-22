from chempy.util.testing import requires
from chempy.units import units_library, default_units as u
from ..numbers import (
def test__float_str_w_uncert():
    assert _float_str_w_uncert(-5739, 16.34, 2) == '-5739(16)'
    assert _float_str_w_uncert(-5739, 16.9, 2) == '-5739(17)'
    assert _float_str_w_uncert(0.0123, 0.00169, 2) == '0.0123(17)'
    assert _float_str_w_uncert(0.01234, 0.00169, 2) == '0.0123(17)'
    assert _float_str_w_uncert(0.01234, 0.0016501, 2) == '0.0123(17)'
    assert _float_str_w_uncert(0.01234, 0.0016501, 1) == '0.012(2)'
    assert _float_str_w_uncert(-9997520000000000.0, 3490000000000.0, 2) == '-9.9975(35)e15'
    assert _float_str_w_uncert(-999752.0, 349, 2) == '-999750(350)'
    assert _float_str_w_uncert(-999752.0, 349, 3) == '-999752(349)'
    assert _float_str_w_uncert(315, 0.00179, 2) == '315.0000(18)'
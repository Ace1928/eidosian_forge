from ..henry import Henry, HenryWithUnits
from ..units import units_library, allclose, default_units as u
from ..util.testing import requires
def test_Henry():
    kH_O2 = Henry(0.0012, 1800, ref='carpenter_1966')
    assert abs(kH_O2(298.15) - 0.0012) < 0.0001
    assert abs(kH_O2.get_c_at_T_and_P(290, 1) - 0.001421892) < 1e-08
    assert abs(kH_O2.get_P_at_T_and_c(310, 0.001) - 1.05) < 0.001
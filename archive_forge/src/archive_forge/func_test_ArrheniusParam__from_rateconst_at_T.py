import math
from chempy.chemistry import Reaction
from chempy.util.testing import requires
from chempy.units import units_library, allclose, default_units as u
from ..arrhenius import arrhenius_equation, ArrheniusParam, ArrheniusParamWithUnits
@requires('numpy')
def test_ArrheniusParam__from_rateconst_at_T():
    ap = ArrheniusParam.from_rateconst_at_T(_Ea1, (_T1, _k1))
    assert abs((ap.A - _A1) / _A1) < 0.0001
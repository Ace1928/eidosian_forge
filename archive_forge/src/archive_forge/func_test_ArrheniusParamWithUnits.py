import math
from chempy.chemistry import Reaction
from chempy.util.testing import requires
from chempy.units import units_library, allclose, default_units as u
from ..arrhenius import arrhenius_equation, ArrheniusParam, ArrheniusParamWithUnits
@requires(units_library)
def test_ArrheniusParamWithUnits():
    _2 = _get_ref2_units()
    ap = ArrheniusParamWithUnits(_2.A, _2.Ea)
    k = ap(_2.T)
    assert abs((k - _2.k) / _2.k) < 0.0001
    r = Reaction({'H2O2': 1}, {'OH': 2}, ap)
    ratc = r.rate_expr().rate_coeff({'temperature': _2.T})
    assert allclose(ratc, _2.k, rtol=0.0001)
import math
from ..eyring import eyring_equation, EyringParam, EyringParamWithUnits
from chempy.util.testing import requires
from chempy.units import allclose, units_library, default_units as u
def test_eyring_equation():
    dH, dS = (40000.0, 100.0)
    T = 123.45
    ref = _kB_over_h * T * math.exp(dS / _R) * math.exp(-dH / _R / T)
    res = eyring_equation(dH, dS, T)
    assert abs((res - ref) / ref) < 1e-10
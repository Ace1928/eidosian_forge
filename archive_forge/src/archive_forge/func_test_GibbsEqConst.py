import math
from chempy.chemistry import Equilibrium
from chempy.util._expr import Expr
from chempy.util.testing import requires
from chempy.units import (
from ..expressions import MassActionEq, GibbsEqConst
def test_GibbsEqConst():
    R, T = (8.314, 298.15)
    dH, dS = (-4000.0, 16)
    gee = GibbsEqConst([dH / R, dS / R])
    ref = math.exp(-(dH - T * dS) / (R * T))
    assert abs((gee({'temperature': T}) - ref) / ref) < 1e-14
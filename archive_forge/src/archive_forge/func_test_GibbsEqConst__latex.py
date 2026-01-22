import math
from chempy.chemistry import Equilibrium
from chempy.util._expr import Expr
from chempy.util.testing import requires
from chempy.units import (
from ..expressions import MassActionEq, GibbsEqConst
@requires('sympy')
def test_GibbsEqConst__latex():
    import sympy
    DH, DS, R, T = sympy.symbols('\\Delta\\ H \\Delta\\ S R T')
    gee = GibbsEqConst([DH / R, DS / R])
    res = gee.eq_const({'temperature': T}, backend=sympy)
    ref = sympy.exp(-(DH - T * DS) / (R * T))
    assert (res - ref).simplify() == 0
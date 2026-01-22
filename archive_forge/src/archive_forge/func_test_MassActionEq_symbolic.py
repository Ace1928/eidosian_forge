import math
from chempy.chemistry import Equilibrium
from chempy.util._expr import Expr
from chempy.util.testing import requires
from chempy.units import (
from ..expressions import MassActionEq, GibbsEqConst
@requires('sympy')
def test_MassActionEq_symbolic():
    import sympy as sp
    K, A, B, C = sp.symbols('K A B C')
    mae = MassActionEq([K])
    eq = Equilibrium({'A'}, {'B', 'C'})
    expr = mae.equilibrium_equation({'A': A, 'B': B, 'C': C}, equilibrium=eq)
    assert expr - K + B * C / A == 0
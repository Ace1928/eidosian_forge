import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
def test_MassAction():
    ma = MassAction([3.14], ['my_rate'])
    assert ma.nargs == 1
    r = Reaction({'A': 2, 'B': 1}, {'C': 1}, ma, {'B': 1})
    arg1 = {'A': 11, 'B': 13, 'C': 17}
    arg2 = {'A': 11, 'B': 13, 'C': 17, 'my_rate': 2.72}
    res1 = r.rate_expr()(arg1, reaction=r)
    res2 = r.rate_expr()(arg2, reaction=r)
    ref1 = 3.14 * 13 * 11 ** 2
    ref2 = 2.72 * 13 * 11 ** 2
    assert abs(res1 - ref1) < 1e-14
    assert abs(res2 - ref2) < 1e-12
    rat1 = r.rate(arg1)
    rat2 = r.rate(arg2)
    for key, coeff in [('A', -2), ('B', -2), ('C', 1)]:
        assert abs(rat1[key] - ref1 * coeff) < 2e-12
        assert abs(rat2[key] - ref2 * coeff) < 2e-12
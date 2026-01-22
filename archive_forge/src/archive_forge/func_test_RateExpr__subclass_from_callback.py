import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
def test_RateExpr__subclass_from_callback():
    SF = RateExpr.subclass_from_callback(lambda v, a, backend: a[0] * v['H2'] * v['Br2'] ** (3 / 2) / (v['Br2'] + a[1] * v['HBr']))
    ratex = SF([11, 13], ['k_HBr', 'kprime_HBr'])
    r = Reaction({'H2': 1, 'Br2': 1}, {'HBr': 2}, ratex)
    res1 = r.rate_expr()({'H2': 5, 'Br2': 7, 'HBr': 15})
    ref1 = 11 * 5 * 7 ** 1.5 / (7 + 13 * 15)
    assert abs((res1 - ref1) / ref1) < 1e-14
    res2 = r.rate_expr()({'H2': 5, 'Br2': 7, 'HBr': 15, 'k_HBr': 23, 'kprime_HBr': 42})
    ref2 = 23 * 5 * 7 ** 1.5 / (7 + 42 * 15)
    assert abs((res2 - ref2) / ref2) < 1e-14
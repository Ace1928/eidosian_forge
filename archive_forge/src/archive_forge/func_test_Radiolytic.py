import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
def test_Radiolytic():
    r = Radiolytic([2.1e-07])
    res = r({'doserate': 0.15, 'density': 0.998})
    assert abs(res - 0.15 * 0.998 * 2.1e-07) < 1e-15
import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
@requires(parsing_library, units_library)
def test_Radiolytic__parsing__units():
    rxn = Reaction.from_string("-> H + OH; Radiolytic({'radiolytic_yield': 2.1e-7*mol/J})", None)
    assert rxn.reac == {}
    assert rxn.prod == {'H': 1, 'OH': 1}
    res = rxn.rate({'doserate': 0.15 * u.gray / u.s, 'density': 0.998 * u.kg / u.dm3})
    ref = 0.15 * 0.998 * 2.1e-07 * u.molar / u.second
    assert abs((res['H'] - ref) / ref) < 1e-15
    assert abs((res['OH'] - ref) / ref) < 1e-15
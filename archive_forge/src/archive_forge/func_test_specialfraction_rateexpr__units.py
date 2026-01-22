import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
@requires(units_library)
def test_specialfraction_rateexpr__units():
    _k, _kprime = (3.5 * u.s ** (-1) * u.molar ** (-0.5), 9.2)
    rsys = _get_SpecialFraction_rsys(_k, _kprime)
    r = rsys.rxns[0]
    conc = {'H2': 2 * u.molar, 'Br2': 3000 * u.mol / u.m ** 3, 'HBr': 5 * u.molar}
    ma = r.rate_expr()

    def _check(k, kprime, c):
        ref = k * c['H2'] * c['Br2'] ** 1.5 / (c['Br2'] + kprime * c['HBr'])
        assert abs(ma(c, reaction=r) - ref) < 1e-15 * u.molar / u.second
    _check(_k, _kprime, conc)
    alt_k, alt_kprime = (2 * u.s ** (-1) * u.molar ** (-0.5), 5)
    _check(alt_k, alt_kprime, dict(list(conc.items()) + [('k_HBr', alt_k), ('kprime_HBr', alt_kprime)]))
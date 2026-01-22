from collections import defaultdict, OrderedDict
from itertools import permutations
import math
import pytest
from chempy import Equilibrium, Reaction, ReactionSystem, Substance
from chempy.thermodynamics.expressions import MassActionEq
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.testing import requires
from .test_rates import _get_SpecialFraction_rsys
from ..arrhenius import ArrheniusParam
from ..rates import Arrhenius, MassAction, Radiolytic, RampedTemp
from .._rates import ShiftedTPoly
from ..ode import (
from ..integrated import dimerization_irrev, binary_rev
@requires('numpy', 'pyodesys', 'sympy')
@pytest.mark.parametrize('substances', permutations(['H2O', 'H+', 'OH-']))
def test_get_odesys__linear_dependencies__preferred(substances):
    rsys = ReactionSystem.from_string('\n'.join(['H2O -> H+ + OH-; 1e-4', 'OH- + H+ -> H2O; 1e10']), substances)
    assert isinstance(rsys.substances, OrderedDict)
    odesys, extra = get_odesys(rsys)
    af_H2O_H = extra['linear_dependencies'](['H+', 'H2O'])
    import sympy
    y0 = {k: sympy.Symbol(k + '0') for k in rsys.substances}
    af_H2O_H(None, {odesys[k]: v for k, v in y0.items()}, None, sympy)
    exprs_H2O_H = af_H2O_H(None, {odesys[k]: v for k, v in y0.items()}, None, sympy)
    ref_H2O_H = {'H2O': y0['H2O'] + y0['OH-'] - odesys['OH-'], 'H+': 2 * y0['H2O'] + y0['H+'] + y0['OH-'] - odesys['OH-'] - 2 * (y0['H2O'] + y0['OH-'] - odesys['OH-'])}
    for k, v in ref_H2O_H.items():
        assert exprs_H2O_H[odesys[k]] - v == 0
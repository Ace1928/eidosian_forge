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
def test_get_odesys__max_euler_step_cb():
    rsys = ReactionSystem.from_string('\n'.join(['H2O -> H+ + OH-; 1e-4', 'OH- + H+ -> H2O; 1e10']))
    odesys, extra = get_odesys(rsys)
    r1 = 0.000101
    r2 = 0.0006
    dH2Odt = r2 - r1
    euler_ref = 2e-07 / dH2Odt
    assert abs(extra['max_euler_step_cb'](0, {'H2O': 1.01, 'H+': 2e-07, 'OH-': 3e-07}) - euler_ref) / euler_ref < 1e-08
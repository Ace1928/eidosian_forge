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
@requires('pyodeint', 'sym', units_library)
def test_get_odesys_rsys_with_units__named_params():
    rsys = ReactionSystem.from_string("\n    A -> B; 'k1'\n    B + C -> P; 'k2'\n    ", substance_factory=Substance)
    odesys, extra = get_odesys(rsys, include_params=False, unit_registry=SI_base_registry)
    tend = 10
    tend_units = tend * u.s
    c0 = {'A': 1e-06, 'B': 0, 'C': 1, 'P': 0}
    p = {'k1': 3, 'k2': 4}
    p_units = {'k1': 3 / u.s, 'k2': 4 / u.M / u.s}
    c0_units = {k: v * u.molar for k, v in c0.items()}
    result1 = odesys.integrate(tend_units, c0_units, p_units, integrator='odeint')
    assert result1.info['success']
    with pytest.raises(Exception):
        odesys.integrate(tend, c0, p, integrator='odeint')
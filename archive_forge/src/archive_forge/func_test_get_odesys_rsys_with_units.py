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
@requires('pygslodeiv2', 'sym', units_library)
def test_get_odesys_rsys_with_units():
    rsys = ReactionSystem.from_string('\n    A -> B; 0.096/s\n    B + C -> P; 4e3/M/s\n    ', substance_factory=Substance)
    with pytest.raises(Exception):
        get_odesys(rsys)
    odesys, extra = get_odesys(rsys, unit_registry=SI_base_registry)
    tend = 10
    tend_units = tend * u.s
    c0 = {'A': 1e-06, 'B': 0, 'C': 1, 'P': 0}
    c0_units = {k: v * u.molar for k, v in c0.items()}
    result1 = odesys.integrate(tend_units, c0_units, integrator='gsl')
    assert result1.info['success']
    with pytest.raises(Exception):
        odesys.integrate(tend, c0, integrator='gsl')
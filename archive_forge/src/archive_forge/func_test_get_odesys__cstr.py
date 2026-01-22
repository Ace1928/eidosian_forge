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
@requires('pyodesys', 'scipy', 'sym')
def test_get_odesys__cstr():
    rsys = ReactionSystem.from_string('2 H2O2 -> O2 + 2 H2O; 5')
    odesys, extra = get_odesys(rsys, cstr=True)
    fr, fc = extra['cstr_fr_fc']
    _check_cstr(odesys, fr, fc)
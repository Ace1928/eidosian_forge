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
@requires('pycvodes', 'sym', units_library)
def test_create_odesys__Radiolytic():
    rsys1 = ReactionSystem.from_string("\n    -> e-(aq); Radiolytic.fk('g_emaq')\n    ", checks=())
    ic1 = {'e-(aq)': 0.0}
    t1 = 5
    p1 = dict(g_emaq=42.0, doserate=17.0, density=5.0)
    odesys1, odesys_extra = create_odesys(rsys1)
    result1 = odesys1.integrate(t1, ic1, p1)
    yref1 = result1.xout * p1['g_emaq'] * p1['doserate'] * p1['density']
    assert np.allclose(yref1, result1.yout.squeeze())
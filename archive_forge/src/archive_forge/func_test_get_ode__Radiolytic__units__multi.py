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
@requires('pyodesys', units_library)
def test_get_ode__Radiolytic__units__multi():
    rad = Radiolytic([2.4e-07 * u.mol / u.joule])
    rxn = Reaction({'A': 4, 'B': 1}, {'C': 3, 'D': 2}, rad)
    rsys = ReactionSystem([rxn], 'A B C D')
    odesys = get_odesys(rsys, include_params=True, unit_registry=SI_base_registry)[0]
    conc = {'A': 3 * u.molar, 'B': 5 * u.molar, 'C': 11 * u.molar, 'D': 13 * u.molar}
    doserates = [dr * u.gray / u.second for dr in [0.1, 0.2, 0.3, 0.4]]
    results = odesys.integrate(37 * u.second, conc, {'doserate': doserates, 'density': 0.998 * u.kg / u.decimetre ** 3})
    assert len(results) == 4
    for i, r in enumerate(results):
        dr = r.params[odesys.param_names.index('doserate')]
        assert dr.ndim == 0 or len(dr) == 1
        assert dr == doserates[i]
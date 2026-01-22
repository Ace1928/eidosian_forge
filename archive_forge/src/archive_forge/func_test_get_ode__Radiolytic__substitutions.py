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
@requires('pyodesys')
def test_get_ode__Radiolytic__substitutions():
    rad = Radiolytic([2.4e-07])
    rxn = Reaction({'A': 4, 'B': 1}, {'C': 3, 'D': 2}, rad)
    rsys = ReactionSystem([rxn], 'A B C D')
    substance_rho = Density([1, -0.001, 273.15])
    odesys = get_odesys(rsys, include_params=True, substitutions={'density': substance_rho})[0]
    conc = {'A': 3, 'B': 5, 'C': 11, 'D': 13}
    state = {'doserate': 0.4, 'temperature': 298.15}
    x, y, p = odesys.to_arrays(-37, conc, state)
    fout = odesys.f_cb(x, y, p)
    r = 2.4e-07 * 0.4 * substance_rho({'temperature': 298.15})
    ref = [-4 * r, -r, 3 * r, 2 * r]
    assert np.all(abs((fout - ref) / ref) < 1e-14)
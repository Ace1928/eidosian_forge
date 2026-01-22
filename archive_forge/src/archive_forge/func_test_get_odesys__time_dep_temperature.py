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
def test_get_odesys__time_dep_temperature():
    import sympy as sp

    def refA(t, A0, A, Ea_over_R, T0, dTdt):
        T = T0 + dTdt * t
        d_Ei = sp.Ei(-Ea_over_R / T0).n(100).round(90) - sp.Ei(-Ea_over_R / T).n(100).round(90)
        d_Texp = T0 * sp.exp(-Ea_over_R / T0) - T * sp.exp(-Ea_over_R / T)
        return A0 * sp.exp(A / dTdt * (Ea_over_R * d_Ei + d_Texp)).n(30)
    params = A0, A, Ea_over_R, T0, dTdt = [13, 10000000000.0, 56000.0 / 8, 273, 2]
    B0 = 11
    rate = MassAction(Arrhenius([A, Ea_over_R]))
    rxn = Reaction({'A': 1}, {'B': 3}, rate)
    rsys = ReactionSystem([rxn], 'A B')
    rt = RampedTemp([T0, dTdt], ('init_temp', 'ramp_rate'))
    odesys, extra = get_odesys(rsys, False, substitutions={'temperature': rt})
    all_pk, unique, p_units = map(extra.get, 'param_keys unique p_units'.split())
    conc = {'A': A0, 'B': B0}
    tout = [2, 5, 10]
    for ramp_rate in [2, 3, 4]:
        unique['ramp_rate'] = ramp_rate
        xout, yout, info = odesys.integrate(10, conc, unique, atol=1e-10, rtol=1e-12)
        params[-1] = ramp_rate
        Aref = np.array([float(refA(t, *params)) for t in xout])
        yref = np.zeros((xout.size, 2))
        yref[:, 0] = Aref
        yref[:, 1] = B0 + 3 * (A0 - Aref)
        assert allclose(yout, yref)
    unique['ramp_rate'] = 2
    x, y, p = odesys.to_arrays(tout, conc, unique)
    fout = odesys.f_cb(x, y, p)

    def r(t):
        return A * np.exp(-Ea_over_R / (T0 + dTdt * t)) * A0
    ref = np.array([[-r(2), -r(5), -r(10)], [3 * r(2), 3 * r(5), 3 * r(10)]]).T
    assert np.allclose(fout, ref)
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
@requires('pycvodes', 'sym', 'scipy', units_library)
def test_get_odesys__Eyring_2nd_order_linearly_ramped_temperature():
    from scipy.special import expi

    def analytic_unit0(t, k, m, dH, dS):
        R = 8.314472
        kB = 1.3806504e-23
        h = 6.62606896e-34
        A = kB / h * np.exp(dS / R)
        B = dH / R
        return k * np.exp(B * (k * t + 2 * m) / (m * (k * t + m))) / (A * (-B ** 2 * np.exp(B / (k * t + m)) * expi(-B / (k * t + m)) - B * k * t - B * m + k ** 2 * t ** 2 + 2 * k * m * t + m ** 2) * np.exp(B / m) + (A * B ** 2 * np.exp(B / m) * expi(-B / m) - A * m * (-B + m) + k * np.exp(B / m)) * np.exp(B / (k * t + m)))
    T_K = 290
    dTdt_Ks = 3
    dH = 80000.0
    dS = 10
    rsys1 = ReactionSystem.from_string('\n    2 NO2 -> N2O4; EyringParam(dH={dH}*J/mol, dS={dS}*J/K/mol)\n    '.format(dH=dH, dS=dS))
    NO2_M = 1.0
    init_cond = dict(NO2=NO2_M * u.M, N2O4=0 * u.M)
    t = 20 * u.second

    def check(rsys):
        odes, extra = get_odesys(rsys, unit_registry=SI_base_registry, constants=const, substitutions={'temperature': RampedTemp([T_K * u.K, dTdt_Ks * u.K / u.s])})
        for odesys in [odes, odes.as_autonomous()]:
            res = odesys.integrate(t, init_cond, integrator='cvode')
            t_sec = to_unitless(res.xout, u.second)
            NO2_ref = analytic_unit0(t_sec, dTdt_Ks, T_K, dH, dS)
            cmp = to_unitless(res.yout, u.M)
            ref = np.empty_like(cmp)
            ref[:, odesys.names.index('NO2')] = NO2_ref
            ref[:, odesys.names.index('N2O4')] = (NO2_M - NO2_ref) / 2
            assert np.allclose(cmp, ref)
    check(rsys1)
    rsys2 = ReactionSystem.from_string('\n    2 NO2 -> N2O4; MassAction(EyringHS([{dH}*J/mol, {dS}*J/K/mol]))\n    '.format(dH=dH, dS=dS))
    check(rsys2)
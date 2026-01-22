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
def test_get_odesys__Eyring_2nd_order():
    R = 8.314472
    T_K = 300
    dH = 80000.0
    dS = 10
    rsys1b = ReactionSystem.from_string('\n    NO + Br -> NOBr; EyringParam(dH={dH}*J/mol, dS={dS}*J/K/mol)\n    '.format(dH=dH, dS=dS))
    c0 = 1
    kbref = 20836643994.118652 * T_K * np.exp(-(dH - T_K * dS) / (R * T_K)) / c0
    NO0_M = 1.5
    Br0_M = 0.7
    init_cond = dict(NOBr=0 * u.M, NO=NO0_M * u.M, Br=Br0_M * u.M)
    t = 5 * u.second
    params = dict(temperature=T_K * u.K)

    def analytic_b(t):
        U, V = (NO0_M, Br0_M)
        d = U - V
        return U * (1 - np.exp(-kbref * t * d)) / (U / V - np.exp(-kbref * t * d))

    def check(rsys):
        odesys, extra = get_odesys(rsys, unit_registry=SI_base_registry, constants=const)
        res = odesys.integrate(t, init_cond, params, integrator='cvode')
        t_sec = to_unitless(res.xout, u.second)
        NOBr_ref = analytic_b(t_sec)
        cmp = to_unitless(res.yout, u.M)
        ref = np.empty_like(cmp)
        ref[:, odesys.names.index('NOBr')] = NOBr_ref
        ref[:, odesys.names.index('Br')] = Br0_M - NOBr_ref
        ref[:, odesys.names.index('NO')] = NO0_M - NOBr_ref
        assert np.allclose(cmp, ref)
    check(rsys1b)
    rsys2b = ReactionSystem.from_string('\n    NO + Br -> NOBr; MassAction(EyringHS([{dH}*J/mol, {dS}*J/K/mol]))\n    '.format(dH=dH, dS=dS))
    check(rsys2b)
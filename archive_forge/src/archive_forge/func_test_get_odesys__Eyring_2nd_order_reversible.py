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
def test_get_odesys__Eyring_2nd_order_reversible():
    R = 8.314472
    T_K = 273.15 + 20
    kB = 1.3806504e-23
    h = 6.62606896e-34
    dHf = 74000.0
    dSf = R * np.log(h / kB / T_K * 1e+16)
    dHb = 79000.0
    dSb = dSf - 23
    rsys1 = ReactionSystem.from_string('\n    Fe+3 + SCN- -> FeSCN+2; EyringParam(dH={dHf}*J/mol, dS={dSf}*J/K/mol)\n    FeSCN+2 -> Fe+3 + SCN-; EyringParam(dH={dHb}*J/mol, dS={dSb}*J/K/mol)\n    '.format(dHf=dHf, dSf=dSf, dHb=dHb, dSb=dSb))
    kf_ref = 20836643994.118652 * T_K * np.exp(-(dHf - T_K * dSf) / (R * T_K))
    kb_ref = 20836643994.118652 * T_K * np.exp(-(dHb - T_K * dSb) / (R * T_K))
    Fe0 = 0.006
    SCN0 = 0.002
    init_cond = {'Fe+3': Fe0 * u.M, 'SCN-': SCN0 * u.M, 'FeSCN+2': 0 * u.M}
    t = 3 * u.second

    def check(rsys, params):
        odes, extra = get_odesys(rsys, include_params=False, unit_registry=SI_base_registry, constants=const)
        for odesys in [odes, odes.as_autonomous()]:
            res = odesys.integrate(t, init_cond, params, integrator='cvode')
            t_sec = to_unitless(res.xout, u.second)
            FeSCN_ref = binary_rev(t_sec, kf_ref, kb_ref, 0, Fe0, SCN0)
            cmp = to_unitless(res.yout, u.M)
            ref = np.empty_like(cmp)
            ref[:, odesys.names.index('FeSCN+2')] = FeSCN_ref
            ref[:, odesys.names.index('Fe+3')] = Fe0 - FeSCN_ref
            ref[:, odesys.names.index('SCN-')] = SCN0 - FeSCN_ref
            assert np.allclose(cmp, ref)
    check(rsys1, {'temperature': T_K * u.K})
    rsys2 = ReactionSystem.from_string('\n    Fe+3 + SCN- -> FeSCN+2; MassAction(EyringHS([{dHf}*J/mol, {dSf}*J/K/mol]))\n    FeSCN+2 -> Fe+3 + SCN-; MassAction(EyringHS([{dHb}*J/mol, {dSb}*J/K/mol]))\n    '.format(dHf=dHf, dSf=dSf, dHb=dHb, dSb=dSb))
    check(rsys2, {'temperature': T_K * u.K})
    rsys3 = ReactionSystem.from_string("\n    Fe+3 + SCN- -> FeSCN+2; MassAction(EyringHS.fk('dHf', 'dSf'))\n    FeSCN+2 -> Fe+3 + SCN-; MassAction(EyringHS.fk('dHb', 'dSb'))\n    ")
    check(rsys3, dict(temperature=T_K * u.K, dHf=dHf * u.J / u.mol, dSf=dSf * u.J / u.mol / u.K, dHb=dHb * u.J / u.mol, dSb=dSb * u.J / u.mol / u.K))
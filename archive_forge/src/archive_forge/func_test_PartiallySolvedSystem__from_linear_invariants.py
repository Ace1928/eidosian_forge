from __future__ import print_function, absolute_import, division
from collections import defaultdict, OrderedDict
from itertools import product
import math
import numpy as np
import pytest
from .. import ODESys
from ..core import integrate_auto_switch, chained_parameter_variation
from ..symbolic import SymbolicSys, ScaledSys, symmetricsys, PartiallySolvedSystem, get_logexp, _group_invariants
from ..util import requires, pycvodes_double, pycvodes_klu
from .bateman import bateman_full  # analytic, never mind the details
from .test_core import vdp_f
from . import _cetsa
@requires('sym', 'pycvodes')
@pytest.mark.parametrize('scaled', [False, True])
def test_PartiallySolvedSystem__from_linear_invariants(scaled):
    atol, rtol, forgive = (1e-11, 1e-11, 20)
    k = [7.0, 3, 2]
    _ss = SymbolicSys.from_callback(decay_rhs, len(k) + 1, len(k), linear_invariants=[[1] * (len(k) + 1)], linear_invariant_names=['tot_amount'])
    if scaled:
        ss = ScaledSys.from_other(_ss, dep_scaling=1000.0)
    else:
        ss = _ss
    y0 = [0] * (len(k) + 1)
    y0[0] = 1

    def check_formulation(odesys):
        xout, yout, info = odesys.integrate([0, 1], y0, k, integrator='cvode', atol=atol, rtol=rtol, nsteps=800)
        ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
        assert np.allclose(yout, ref, rtol=rtol * forgive, atol=atol * forgive)
    check_formulation(ss)
    ps = PartiallySolvedSystem.from_linear_invariants(ss)
    assert ps.ny == ss.ny - 1
    check_formulation(ps)
    ps2 = PartiallySolvedSystem(ss, lambda x0, y0, p0, be: {ss.dep[0]: y0[0] * be.exp(-p0[0] * (ss.indep - x0))})
    assert ps2.ny == ss.ny - 1
    check_formulation(ps2)
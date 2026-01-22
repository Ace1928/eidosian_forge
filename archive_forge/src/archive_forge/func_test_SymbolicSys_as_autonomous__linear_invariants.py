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
def test_SymbolicSys_as_autonomous__linear_invariants():

    def rhs(t, y, p):
        k = t ** p[0]
        return [-k * y[0], k * y[0]]

    def analytic(tout, init_y, params):
        y0ref = init_y[0] * np.exp(-tout ** (params[0] + 1) / (params[0] + 1))
        return np.array([y0ref, init_y[0] - y0ref + init_y[1]]).T
    odes = SymbolicSys.from_callback(rhs, 2, 1, linear_invariants=[[1, 1]])
    for odesys in [odes, odes.as_autonomous()]:
        result = odesys.integrate(4, [5, 2], [3], integrator='cvode')
        ref = analytic(result.xout, result.yout[0, :], result.params)
        assert np.allclose(result.yout, ref, atol=1e-06)
        invar_viol = result.calc_invariant_violations()
        assert np.allclose(invar_viol, 0)
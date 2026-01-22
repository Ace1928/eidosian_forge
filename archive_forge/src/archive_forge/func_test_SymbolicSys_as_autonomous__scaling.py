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
def test_SymbolicSys_as_autonomous__scaling():

    def get_odesys(scaling=1):

        def rhs(t, y, p, backend=math):
            HNO2, H2O, NO, NO2, N2O4 = y
            dH1, dS1, dH2, dS2 = p
            R = 8.314
            T = 300 + 10 * backend.sin(0.2 * math.pi * t - math.pi / 2)
            kB_h = 20836600000.0
            k1 = kB_h * T * backend.exp(dS1 / R - dH1 / (R * T)) / scaling
            k2 = kB_h * T * backend.exp(dS2 / R - dH2 / (R * T)) / scaling
            r1 = k1 * HNO2 ** 2
            r2 = k2 * NO2 ** 2
            return [-2 * r1, r1, r1, r1 - 2 * r2, r2]
        return SymbolicSys.from_callback(rhs, 5, 4, names='HNO2 H2O NO NO2 N2O4'.split(), param_names='dH1 dS1 dH2 dS2'.split())

    def check(system, scaling=1):
        init_y = [1 * scaling, 55 * scaling, 0, 0, 0]
        p = [85000.0, 10, 70000.0, 20]
        return system.integrate(np.linspace(0, 60, 200), init_y, p, integrator='cvode', nsteps=5000)

    def compare_autonomous(scaling):
        odesys = get_odesys(scaling)
        autsys = odesys.as_autonomous()
        copsys = SymbolicSys.from_other(autsys)
        res1 = check(odesys, scaling=scaling)
        res2 = check(autsys, scaling=scaling)
        res3 = check(copsys, scaling=scaling)
        assert np.allclose(res1.yout, res2.yout, atol=1e-06)
        assert np.allclose(res1.yout, res3.yout, atol=1e-06)
    compare_autonomous(1)
    compare_autonomous(1000)
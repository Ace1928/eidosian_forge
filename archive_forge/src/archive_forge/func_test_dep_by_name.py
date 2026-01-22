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
def test_dep_by_name():

    def _sin(t, y, p):
        return {'prim': y['bis'], 'bis': -p[0] ** 2 * y['prim']}
    odesys = SymbolicSys.from_callback(_sin, names=['prim', 'bis'], nparams=1, dep_by_name=True)
    A, k = (2, 3)
    for y0 in ({'prim': 0, 'bis': A * k}, [0, A * k]):
        xout, yout, info = odesys.integrate(np.linspace(0, 1), y0, [k], integrator='cvode', method='adams')
        assert info['success']
        assert xout.size > 7
        ref = [A * np.sin(k * (xout - xout[0])), A * np.cos(k * (xout - xout[0])) * k]
        assert np.allclose(yout[:, 0], ref[0], atol=1e-05, rtol=1e-05)
        assert np.allclose(yout[:, 1], ref[1], atol=1e-05, rtol=1e-05)
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
@pycvodes_klu
def test_SymbolicSys_jacobian_sparse():
    k = (4, 3)
    y0 = (5, 4, 2)
    odesys = SymbolicSys.from_callback(decay_dydt_factory(k), len(k) + 1, sparse=True)
    xout, yout, info = odesys.integrate([0, 2], y0, integrator='cvode', linear_solver='klu', atol=1e-12, rtol=1e-12, nsteps=1000)
    ref = np.array(bateman_full(y0, k + (0,), xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=1e-10, atol=1e-10)
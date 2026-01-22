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
@requires('sym', 'scipy')
@pytest.mark.parametrize('name,forgive', zip('dopri5 dop853 vode'.split(), (1, 1, (3, 3000000.0))))
def test_scipy(name, forgive):
    n, p, a = (13, 1, 13)
    atol, rtol = (1e-10, 1e-10)
    y0, k, odesys_dens = get_special_chain(n, p, a)
    if name == 'vode':
        tout = [0] + [10 ** i for i in range(-10, 1)]
        xout, yout, info = odesys_dens.integrate(tout, y0, integrator='scipy', name=name, atol=atol, rtol=rtol)
        check(yout[-1, :], n, p, a, atol, rtol, forgive[0])
        xout, yout, info = odesys_dens.integrate(1, y0, integrator='scipy', name=name, atol=atol, rtol=rtol)
        check(yout[-1, :], n, p, a, atol, rtol, forgive[1])
    else:
        xout, yout, info = odesys_dens.integrate(1, y0, integrator='scipy', name=name, atol=atol, rtol=rtol)
        check(yout[-1, :], n, p, a, atol, rtol, forgive)
    assert yout.shape[0] > 2
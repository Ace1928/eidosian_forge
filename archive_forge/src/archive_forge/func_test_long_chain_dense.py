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
@pytest.mark.parametrize('n,forgive', [(4, 1), (17, 1), (42, 7)])
def test_long_chain_dense(n, forgive):
    p, a = (0, n)
    y0, k, odesys_dens = get_special_chain(n, p, a)
    atol, rtol = (1e-12, 1e-12)
    tout = 1
    xout, yout, info = odesys_dens.integrate(tout, y0, integrator='scipy', atol=atol, rtol=rtol)
    check(yout[-1, :], n, p, a, atol, rtol, forgive)
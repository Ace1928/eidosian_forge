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
@requires('sym', 'pyodeint')
def test_PartiallySolvedSystem__by_name__from_linear_invariants():
    k = [math.log(2) / (138.4 * 24 * 3600)]
    names = 'Po-210 Pb-206'.split()
    odesys = SymbolicSys.from_callback(decay_dydt_factory({'Po-210': k[0]}, names=names), dep_by_name=True, names=names, linear_invariants=[[1, 1]])
    assert odesys.ny == 2
    partsys1 = PartiallySolvedSystem.from_linear_invariants(odesys)
    partsys2 = PartiallySolvedSystem.from_linear_invariants(odesys, ['Pb-206'])
    partsys3 = PartiallySolvedSystem.from_linear_invariants(odesys, ['Po-210'])
    assert partsys1.free_names in (['Po-210'], ['Pb-206'])
    assert partsys2.free_names == ['Po-210']
    assert partsys3.free_names == ['Pb-206']
    assert partsys1.ny == partsys2.ny == partsys3.ny == 1
    assert partsys2['Pb-206'] - partsys2.init_dep[partsys2.names.index('Pb-206')] - partsys2.init_dep[partsys2.names.index('Po-210')] + odesys['Po-210'] == 0
    duration = 7 * k[0]
    atol, rtol, forgive = (1e-09, 1e-09, 10)
    y0 = [1e-20] * (len(k) + 1)
    y0[0] = 1
    for system in (odesys, partsys1, partsys2, partsys3):
        xout, yout, info = system.integrate(duration, y0, integrator='odeint', rtol=rtol, atol=atol)
        ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
        assert np.allclose(yout, ref, rtol=rtol * forgive, atol=atol * forgive)
        assert yout.shape[1] == 2
        assert xout.shape[0] == yout.shape[0]
        assert yout.ndim == 2 and xout.ndim == 1
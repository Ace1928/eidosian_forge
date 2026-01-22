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
@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_PartiallySolvedSystem__symmetricsys__nonlinear(integrator):
    partsys = _get_nonlin_part_system()
    logexp = get_logexp(7, partsys.indep ** 0 / 10 ** 7)
    trnsfsys = symmetricsys(logexp, logexp).from_other(partsys)
    y0 = [3.0, 2.0, 1.0]
    k = [9.351, 2.532]
    tend = 1.7
    atol, rtol = (1e-12, 1e-13)
    for odesys, forgive in [(partsys, 21), (trnsfsys, 298)]:
        xout, yout, info = odesys.integrate(tend, y0, k, integrator=integrator, first_step=1e-14, atol=atol, rtol=rtol, nsteps=1000)
        assert info['success']
        yref = np.empty_like(yout)
        yref[:, 2] = _ref_nonlin(y0, k, xout - xout[0])
        yref[:, 0] = y0[0] + y0[2] - yref[:, 2]
        yref[:, 1] = y0[1] + y0[2] - yref[:, 2]
        assert np.allclose(yout, yref, atol=forgive * atol, rtol=forgive * rtol)
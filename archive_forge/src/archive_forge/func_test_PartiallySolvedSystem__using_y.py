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
def test_PartiallySolvedSystem__using_y(integrator):
    odesys = _get_decay3()
    partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0: {odesys.dep[2]: y0[0] + y0[1] + y0[2] - odesys.dep[0] - odesys.dep[1]})
    y0 = [3, 2, 1]
    k = [3.5, 2.5, 0]
    xout, yout, info = partsys.integrate([0, 1], y0, k, integrator=integrator)
    ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
    assert info['success']
    assert np.allclose(yout, ref)
    assert np.allclose(np.sum(yout, axis=1), sum(y0))
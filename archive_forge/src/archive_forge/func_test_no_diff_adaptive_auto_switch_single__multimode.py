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
def test_no_diff_adaptive_auto_switch_single__multimode(integrator):
    odesys = _get_decay3()
    tout = [[3, 5], [4, 6], [6, 8], [9, 11]]
    _y0 = [3, 2, 1]
    y0 = [_y0] * 4
    _k = [3.5, 2.5, 1.5]
    k = [_k] * 4
    res1 = odesys.integrate(tout, y0, k, integrator=integrator, first_step=1e-14)
    for res in res1:
        xout1, yout1, info1 = (res.xout, res.yout, res.info)
        ref = np.array(bateman_full(_y0, _k, xout1 - xout1[0], exp=np.exp)).T
        assert info1['success']
        assert xout1.size > 10
        assert xout1.size == yout1.shape[0]
        assert np.allclose(yout1, ref)
    res2 = integrate_auto_switch([odesys], {}, tout, y0, k, integrator=integrator, first_step=1e-14)
    for res in res2:
        xout2, yout2, info2 = (res.xout, res.yout, res.info)
        assert info2['success']
        assert xout2.size == xout1.size
        assert np.allclose(yout2, ref)
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
def test_SymbolicSys_from_other_new_params():
    odesys = _get_decay3()
    assert len(odesys.params) == 3
    p0, p1, p2 = odesys.params
    p3, = odesys.be.real_symarray('p', len(odesys.params) + 1)[-1:]
    newode, extra = SymbolicSys.from_other_new_params(odesys, OrderedDict([(p0, p3 - 1), (p1, p3 + 1)]), (p3,))
    assert len(newode.params) == 2
    tout = np.array([0.3, 0.4, 0.7, 0.9, 1.3, 1.7, 1.8, 2.1])
    y0 = [7, 5, 2]
    p_vals = [5, 3]
    res1 = newode.integrate(tout, y0, p_vals)
    k = [p_vals[1] - 1, p_vals[1] + 1, p_vals[0]]
    ref1 = np.array(bateman_full(y0, k, res1.xout - res1.xout[0], exp=np.exp)).T
    assert np.allclose(res1.yout, ref1)
    orip = extra['recalc_params'](res1.xout, res1.yout, res1.params)
    assert np.allclose(orip, np.atleast_2d([3 - 1, 3 + 1, 5]))
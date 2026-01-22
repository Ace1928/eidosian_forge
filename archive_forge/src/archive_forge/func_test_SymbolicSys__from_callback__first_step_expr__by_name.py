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
def test_SymbolicSys__from_callback__first_step_expr__by_name():
    kwargs = dict(integrator='cvode', atol=1e-08, rtol=1e-08)
    names = ['foo', 'bar', 'baz']
    par_names = 'first second third'.split()
    odesys = SymbolicSys.from_callback(lambda x, y, p, be: {'foo': -p['first'] * y['foo'], 'bar': p['first'] * y['foo'] - p['second'] * y['bar'], 'baz': p['second'] * y['bar'] - p['third'] * y['baz']}, names=names, param_names=par_names, dep_by_name=True, par_by_name=True, first_step_factory=lambda x0, ic: 1e-30 * ic['foo'])
    y0 = {'foo': 0.7, 'bar': 0, 'baz': 0}
    p = {'first': 1e+23, 'second': 2, 'third': 3}
    result = odesys.integrate(5, y0, p, **kwargs)
    assert result.info['success']
    ref = bateman_full([y0[k] for k in names], [p[k] for k in par_names], result.xout - result.xout[0], exp=np.exp)
    for i, k in enumerate(odesys.names):
        assert np.allclose(result.named_dep(k), ref[i], atol=10 * kwargs['atol'], rtol=10 * kwargs['rtol'])
    for k, v in p.items():
        assert result.named_param(k) == v
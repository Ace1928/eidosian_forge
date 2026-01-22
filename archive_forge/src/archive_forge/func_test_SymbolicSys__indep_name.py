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
def test_SymbolicSys__indep_name():
    odesys = SymbolicSys.from_callback(lambda t, y, p: {'x': -p['a'] * y['x'], 'y': -p['b'] * y['y'] + p['a'] * y['x'], 'z': p['b'] * y['y']}, names='xyz', param_names='ab', dep_by_name=True, par_by_name=True)
    pars = {'a': [11, 17, 19], 'b': 13}
    results = odesys.integrate([42, 43, 44], {'x': 7, 'y': 5, 'z': 3}, pars)
    for r, a in zip(results, pars['a']):
        assert np.allclose(r.named_dep('x'), 7 * np.exp(-a * (r.xout - r.xout[0])))
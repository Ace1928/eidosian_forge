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
def test_ScaledSys_from_callback__exprs():

    def f(t, x, k):
        return [-k[0] * x[0] * x[0] * t]
    x, y, nfo = SymbolicSys.from_callback(f, 1, 1).integrate([0, 1], [3.14], [2.78])
    xs, ys, nfos = ScaledSys.from_callback(f, 1, 1, 100).integrate([0, 1], [3.14], [2.78])
    from scipy.interpolate import interp1d
    cb = interp1d(x, y[:, 0])
    cbs = interp1d(xs, ys[:, 0])
    t = np.linspace(0, 1)
    assert np.allclose(cb(t), cbs(t))
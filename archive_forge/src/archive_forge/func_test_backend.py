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
def test_backend():

    def f(x, y, p, backend=math):
        return [backend.exp(p[0] * y[0])]

    def analytic(x, p, y0):
        return -np.log(p * (np.exp(-p * y0) / p - x)) / p
    y0, tout, p = (0.07, [0, 0.1, 0.2], 0.3)
    ref = analytic(tout, p, y0)

    def _test_odesys(odesys):
        yout, info = odesys.predefined([y0], tout, [p])
        assert np.allclose(yout.flatten(), ref)
    _test_odesys(ODESys(f))
    _test_odesys(SymbolicSys.from_callback(f, 1, 1))
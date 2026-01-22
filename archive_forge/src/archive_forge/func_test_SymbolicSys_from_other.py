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
def test_SymbolicSys_from_other():
    scaled = ScaledSys.from_callback(lambda x, y: [y[0] * y[0]], 1, dep_scaling=101)
    LogLogSys = symmetricsys(get_logexp(), get_logexp())
    transformed_scaled = LogLogSys.from_other(scaled)
    tout = np.array([0, 0.2, 0.5])
    y0 = [1.0]
    ref, nfo1 = ODESys(lambda x, y: y[0] * y[0]).predefined(y0, tout, first_step=1e-14)
    analytic = 1 / (1 - tout.reshape(ref.shape))
    assert np.allclose(ref, analytic)
    yout, nfo0 = transformed_scaled.predefined(y0, tout + 1)
    assert np.allclose(yout, analytic)
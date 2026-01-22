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
def integrate_and_check(system):
    init_y = [0, 0]
    p = [2]
    result = system.integrate([0, 80], init_y, p, integrator='cvode', nsteps=5000)
    yref = analytic(result.xout, init_y, p)
    assert np.all(result.yout - yref < 1.6e-05)
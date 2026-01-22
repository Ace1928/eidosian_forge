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
def compare_autonomous(scaling):
    odesys = get_odesys(scaling)
    autsys = odesys.as_autonomous()
    copsys = SymbolicSys.from_other(autsys)
    res1 = check(odesys)
    res2 = check(autsys)
    res3 = check(copsys)
    assert np.allclose(res1.yout, res2.yout, atol=1e-06)
    assert np.allclose(res1.yout, res3.yout, atol=1e-06)
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
@pytest.mark.parametrize('n', [29, 79])
def test_long_chain_banded_cvode(n):
    p, a = (0, n)
    y0, k, odesys_dens = get_special_chain(n, p, a)
    y0, k, odesys_band = get_special_chain(n, p, a, band=(1, 0))
    atol, rtol = (1e-09, 1e-09)

    def mk_callback(odesys):

        def callback(*args, **kwargs):
            return odesys.integrate(*args, integrator='cvode', **kwargs)
        return callback
    for _ in range(2):
        time_band, (xout_band, yout_band, info) = timeit(mk_callback(odesys_band), 1, y0, atol=atol, rtol=rtol)
        assert info['njev'] > 0
    for _ in range(2):
        time_dens, (xout_dens, yout_dens, info) = timeit(mk_callback(odesys_dens), 1, y0, atol=atol, rtol=rtol)
        assert info['njev'] > 0
    check(yout_dens[-1, :], n, p, a, atol, rtol, 7)
    check(yout_band[-1, :], n, p, a, atol, rtol, 25)
    assert info['njev'] > 0
    try:
        assert time_dens > time_band
    except AssertionError:
        pass
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
@requires('sym', 'scipy', 'pycvodes')
@pytest.mark.parametrize('nbody', [2, 3, 4, 5])
def test_ScaledSysByName(nbody):
    sfact = nbody * 7
    kwargs = dict(names=['foo', 'bar'], dep_scaling=sfact)

    def nmerization(x, y, p):
        return [-nbody * p[0] * y[0] ** nbody, nbody * p[0] * y[0] ** nbody]
    odesys = ScaledSys.from_callback(nmerization, 2, 1, **kwargs)
    assert odesys.autonomous_interface is True
    with pytest.raises(TypeError):
        odesys.integrate(1, [0])

    def nmerization_name(x, y, p):
        return {'foo': -nbody * p[0] * y['foo'] ** nbody, 'bar': nbody * p[0] * y['foo'] ** nbody}
    odesys2 = ScaledSys.from_callback(nmerization_name, dep_by_name=True, nparams=1, **kwargs)
    assert odesys2.autonomous_interface is True
    k = 5
    foo0 = 2
    for system, y0 in zip([odesys, odesys2], [[foo0, 3], {'foo': foo0, 'bar': 3}]):
        xout, yout, info = system.integrate(1, y0, [k], integrator='cvode', nsteps=707 * 1.01, first_step=0.001, atol=1e-10, rtol=1e-10)
        _r = (1 / (foo0 ** (1 - nbody) + nbody * k * xout * (nbody - 1))) ** (1 / (nbody - 1))
        assert np.allclose(yout[:, 0], _r, atol=1e-09, rtol=1e-09)
        assert np.allclose(yout[:, 1], 3 + 2 - _r, atol=1e-09, rtol=1e-09)
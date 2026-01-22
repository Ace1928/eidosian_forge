from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('pyodeint', 'scipy')
def test_par_by_name__multi():
    from scipy.special import binom
    for ny in range(6, 8):
        p_max = 3
        a = 0.42
        params = OrderedDict([(chr(ord('a') + idx), [(idx + 1 + p) * math.log(a + 1) for p in range(p_max + 1)]) for idx in range(ny)])
        ref = np.array([[binom(p + idx, p) * (a / (a + 1)) ** idx / (a + 1) ** (p + 1) for idx in range(ny)] for p in range(p_max + 1)])
        odesys = ODESys(*decay_factory(ny), param_names=params.keys(), par_by_name=True)
        results = odesys.integrate(np.linspace(0, 1), [1] + [0] * (ny - 1), params, integrator='odeint', method='rosenbrock4')
        assert all((r.info['success'] for r in results))
        assert all((r.xout.shape[-1] == 50 for r in results))
        assert all((np.allclose(r.yout[-1, :], ref[i, ...]) for i, r in enumerate(results)))
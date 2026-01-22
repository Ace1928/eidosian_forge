from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('scipy', 'pygslodeiv2', 'pycvodes', 'pyodeint')
@pytest.mark.parametrize('integrator', ['scipy', 'gsl', 'cvode', 'odeint'])
def test_adaptive(integrator):
    odes = ODESys(vdp_f, vdp_j, dfdx=vdp_dfdt)
    kwargs = dict(params=[2.0])
    y0, t0, tend = ([1, 0], 0, 2)
    xout, yout, info = odes.adaptive(y0, t0, tend, integrator=integrator, **kwargs)
    ref = [-1.89021896, -0.71633577]
    assert xout.size > 1
    assert np.allclose(yout[-1, :], ref, rtol=0.2 if integrator == 'odeint' else 1e-05)
    assert info['success']
    xout2, yout2, info2 = integrate_chained([odes], {}, [t0, tend], y0, **kwargs)
    assert xout2.size > 1
    assert np.allclose(ref, yout2[-1, :])
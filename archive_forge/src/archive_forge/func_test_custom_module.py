from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
def test_custom_module():
    from pyodesys.integrators import RK4_example_integrator
    odes = ODESys(vdp_f, vdp_j)
    xout, yout, info = odes.integrate([0, 2], [1, 0], params=[2.0], integrator=RK4_example_integrator, first_step=0.01)
    assert np.allclose(yout[0], [1, 0])
    assert np.allclose(yout[-1], [-1.89021896, -0.71633577])
    assert info['nfev'] == 4 * 2 / 0.01
    xout, yout, info = odes.integrate(np.linspace(0, 2, 150), [1, 0], params=[2.0], integrator=RK4_example_integrator)
    assert np.allclose(yout[0], [1, 0])
    assert np.allclose(yout[-1], [-1.89021896, -0.71633577])
    assert info['nfev'] == 4 * 149
from __future__ import (absolute_import, division, print_function)
from ..util import requires
from pyodesys import ODESys
from .test_core import vdp_f, vdp_j
@requires('matplotlib')
def test_plot_result():
    odes = ODESys(vdp_f, vdp_j)
    odes.integrate([0, 1, 2], [1, 0], params=[2.0], integrator='scipy')
    odes.plot_result()
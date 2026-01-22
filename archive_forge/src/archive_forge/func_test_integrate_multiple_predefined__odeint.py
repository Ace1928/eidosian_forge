from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('pyodeint')
def test_integrate_multiple_predefined__odeint():
    _test_integrate_multiple_predefined(ODESys(decay), integrator='odeint', method='bulirsch_stoer', atol=1e-10, rtol=1e-10)
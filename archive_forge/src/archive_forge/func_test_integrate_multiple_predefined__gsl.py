from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('pygslodeiv2')
def test_integrate_multiple_predefined__gsl():
    _test_integrate_multiple_predefined(ODESys(decay), integrator='gsl', method='rkck')
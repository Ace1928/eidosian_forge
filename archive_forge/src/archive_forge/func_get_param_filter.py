import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def get_param_filter(self):
    if self.endpt_rtol is None and self.endpt_atol is None:
        return None
    filters = []
    for rtol, atol, spec in zip(self.endpt_rtol, self.endpt_atol, self.argspec):
        if rtol is None and atol is None:
            filters.append(None)
            continue
        elif rtol is None:
            rtol = 0.0
        elif atol is None:
            atol = 0.0
        filters.append(EndpointFilter(spec.a, spec.b, rtol, atol))
    return filters
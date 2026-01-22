from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def test_solve_series():
    neqsys = NeqSys(1, 1, lambda x, p: [x[0] - p[0]])
    xout, sols = neqsys.solve_series([0], [0], [0, 1, 2, 3], 0, solver='scipy')
    assert np.allclose(xout[:, 0], [0, 1, 2, 3])
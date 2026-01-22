import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_ind(self):
    with np.errstate(divide='ignore', invalid='ignore'):
        assert_all(np.isinf(np.array((0.0,)) / 0.0) == 0)
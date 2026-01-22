import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_posinf(self):
    with np.errstate(divide='ignore', invalid='ignore'):
        assert_all(np.isinf(np.array((1.0,)) / 0.0) == 1)
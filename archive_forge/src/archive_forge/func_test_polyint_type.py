import os
import numpy as np
from numpy.testing import (
def test_polyint_type(self):
    msg = 'Wrong type, should be complex'
    x = np.ones(3, dtype=complex)
    assert_(np.polyint(x).dtype == complex, msg)
    msg = 'Wrong type, should be float'
    x = np.ones(3, dtype=int)
    assert_(np.polyint(x).dtype == float, msg)
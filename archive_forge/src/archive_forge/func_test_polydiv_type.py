import os
import numpy as np
from numpy.testing import (
def test_polydiv_type(self):
    msg = 'Wrong type, should be complex'
    x = np.ones(3, dtype=complex)
    q, r = np.polydiv(x, x)
    assert_(q.dtype == complex, msg)
    msg = 'Wrong type, should be float'
    x = np.ones(3, dtype=int)
    q, r = np.polydiv(x, x)
    assert_(q.dtype == float, msg)
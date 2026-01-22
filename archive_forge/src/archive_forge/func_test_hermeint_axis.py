from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermeint_axis(self):
    c2d = np.random.random((3, 4))
    tgt = np.vstack([herme.hermeint(c) for c in c2d.T]).T
    res = herme.hermeint(c2d, axis=0)
    assert_almost_equal(res, tgt)
    tgt = np.vstack([herme.hermeint(c) for c in c2d])
    res = herme.hermeint(c2d, axis=1)
    assert_almost_equal(res, tgt)
    tgt = np.vstack([herme.hermeint(c, k=3) for c in c2d])
    res = herme.hermeint(c2d, k=3, axis=1)
    assert_almost_equal(res, tgt)
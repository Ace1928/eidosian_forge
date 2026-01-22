from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermefromroots(self):
    res = herme.hermefromroots([])
    assert_almost_equal(trim(res), [1])
    for i in range(1, 5):
        roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
        pol = herme.hermefromroots(roots)
        res = herme.hermeval(roots, pol)
        tgt = 0
        assert_(len(pol) == i + 1)
        assert_almost_equal(herme.herme2poly(pol)[-1], 1)
        assert_almost_equal(res, tgt)
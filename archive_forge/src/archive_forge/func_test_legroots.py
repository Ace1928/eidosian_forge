from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legroots(self):
    assert_almost_equal(leg.legroots([1]), [])
    assert_almost_equal(leg.legroots([1, 2]), [-0.5])
    for i in range(2, 5):
        tgt = np.linspace(-1, 1, i)
        res = leg.legroots(leg.legfromroots(tgt))
        assert_almost_equal(trim(res), trim(tgt))
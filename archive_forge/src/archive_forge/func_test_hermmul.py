from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermmul(self):
    for i in range(5):
        pol1 = [0] * i + [1]
        val1 = herm.hermval(self.x, pol1)
        for j in range(5):
            msg = f'At i={i}, j={j}'
            pol2 = [0] * j + [1]
            val2 = herm.hermval(self.x, pol2)
            pol3 = herm.hermmul(pol1, pol2)
            val3 = herm.hermval(self.x, pol3)
            assert_(len(pol3) == i + j + 1, msg)
            assert_almost_equal(val3, val1 * val2, err_msg=msg)
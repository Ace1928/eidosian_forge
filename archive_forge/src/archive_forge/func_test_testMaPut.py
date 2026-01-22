from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testMaPut(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf, s = self.d
    m = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    i = np.nonzero(m)[0]
    put(ym, i, zm)
    assert_(all(take(ym, i, axis=0) == zm))
from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_linear_root(self):
    assert_(leg.legcompanion([1, 2])[0, 0] == -0.5)
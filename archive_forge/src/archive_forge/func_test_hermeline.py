from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermeline(self):
    assert_equal(herme.hermeline(3, 4), [3, 4])
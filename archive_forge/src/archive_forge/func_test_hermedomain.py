from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermedomain(self):
    assert_equal(herme.hermedomain, [-1, 1])
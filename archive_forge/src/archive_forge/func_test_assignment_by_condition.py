from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_assignment_by_condition(self):
    a = array([1, 2, 3, 4], mask=[1, 0, 1, 0])
    c = a >= 3
    a[c] = 5
    assert_(a[2] is masked)
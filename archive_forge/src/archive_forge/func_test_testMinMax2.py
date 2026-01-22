from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testMinMax2(self):
    assert_(eq(minimum([1, 2, 3], [4, 0, 9]), [1, 0, 3]))
    assert_(eq(maximum([1, 2, 3], [4, 0, 9]), [4, 2, 9]))
    x = arange(5)
    y = arange(5) - 2
    x[3] = masked
    y[0] = masked
    assert_(eq(minimum(x, y), where(less(x, y), x, y)))
    assert_(eq(maximum(x, y), where(greater(x, y), x, y)))
    assert_(minimum.reduce(x) == 0)
    assert_(maximum.reduce(x) == 4)
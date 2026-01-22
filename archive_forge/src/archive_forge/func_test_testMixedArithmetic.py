from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testMixedArithmetic(self):
    na = np.array([1])
    ma = array([1])
    assert_(isinstance(na + ma, MaskedArray))
    assert_(isinstance(ma + na, MaskedArray))
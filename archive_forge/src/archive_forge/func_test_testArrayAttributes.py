from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testArrayAttributes(self):
    a = array([1, 3, 2])
    assert_equal(a.ndim, 1)
import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_corecursive_input(self):
    a, b = ([], [])
    a.append(b)
    b.append(a)
    obj_arr = np.array([None])
    obj_arr[0] = a
    assert_raises(ValueError, obj_arr.astype, 'M8')
    assert_raises(ValueError, obj_arr.astype, 'm8')
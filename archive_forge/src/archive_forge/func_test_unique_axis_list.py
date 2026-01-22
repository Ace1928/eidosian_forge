import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_unique_axis_list(self):
    msg = 'Unique failed on list of lists'
    inp = [[0, 1, 0], [0, 1, 0]]
    inp_arr = np.asarray(inp)
    assert_array_equal(unique(inp, axis=0), unique(inp_arr, axis=0), msg)
    assert_array_equal(unique(inp, axis=1), unique(inp_arr, axis=1), msg)
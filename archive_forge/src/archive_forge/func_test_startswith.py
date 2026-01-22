import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_startswith(self):
    assert_(issubclass(self.A.startswith('').dtype.type, np.bool_))
    assert_array_equal(self.A.startswith(' '), [[1, 0], [0, 0], [0, 0]])
    assert_array_equal(self.A.startswith('1', 0, 3), [[0, 0], [1, 0], [1, 0]])

    def fail():
        self.A.startswith('3', 'fdjk')
    assert_raises(TypeError, fail)
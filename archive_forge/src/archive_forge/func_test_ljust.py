import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_ljust(self):
    assert_(issubclass(self.A.ljust(10).dtype.type, np.bytes_))
    C = self.A.ljust([10, 20])
    assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])
    C = self.A.ljust(20, b'#')
    assert_array_equal(C.startswith(b'#'), [[False, True], [False, False], [False, False]])
    assert_(np.all(C.endswith(b'#')))
    C = np.char.ljust(b'FOO', [[10, 20], [15, 8]])
    tgt = [[b'FOO       ', b'FOO                 '], [b'FOO            ', b'FOO     ']]
    assert_(issubclass(C.dtype.type, np.bytes_))
    assert_array_equal(C, tgt)
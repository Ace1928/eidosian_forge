import pytest
import numpy as np
from numpy.testing import assert_, assert_raises
def test_char_repeat(self):
    np_s = np.bytes_('abc')
    np_u = np.str_('abc')
    res_s = b'abc' * 5
    res_u = 'abc' * 5
    assert_(np_s * 5 == res_s)
    assert_(np_u * 5 == res_u)
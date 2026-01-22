import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
@pytest.mark.parametrize('dtype', ['c', 'S1'])
def test_2d_array_inout(self, dtype):
    f = getattr(self.module, self.fprefix + '_2d_array_inout')
    n = np.array([['A', 'B', 'C'], ['D', 'E', 'F']], dtype=dtype, order='F')
    a = np.array([['a', 'b', 'c'], ['d', 'e', 'f']], dtype=dtype, order='F')
    f(a, n)
    assert_array_equal(a, n)
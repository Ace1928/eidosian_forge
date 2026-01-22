import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
@pytest.mark.parametrize('dtype', ['c', 'S1', 'U1'])
def test_2d_array_input(self, dtype):
    f = getattr(self.module, self.fprefix + '_2d_array_input')
    a = np.array([['a', 'b', 'c'], ['d', 'e', 'f']], dtype=dtype, order='F')
    expected = a.view(np.uint32 if dtype == 'U1' else np.uint8)
    assert_array_equal(f(a), expected)
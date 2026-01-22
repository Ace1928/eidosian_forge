import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
@pytest.mark.parametrize('state', ['new', 'old'])
def test_character_bc(self, state):
    f = getattr(self.module, self.fprefix + '_character_bc_' + state)
    c, a = f()
    assert_equal(c, b'a')
    assert_equal(len(a), 1)
    c, a = f(b'b')
    assert_equal(c, b'b')
    assert_equal(len(a), 2)
    assert_raises(Exception, lambda: f(b'c'))
import gc
from scipy._lib._gcutils import (set_gc_state, gc_state, assert_deallocated,
from numpy.testing import assert_equal
import pytest
@pytest.mark.skipif(IS_PYPY, reason='Test not meaningful on PyPy')
def test_assert_deallocated():

    class C:

        def __init__(self, arg0, arg1, name='myname'):
            self.name = name
    for gc_current in (True, False):
        with gc_state(gc_current):
            with assert_deallocated(C, 0, 2, 'another name') as c:
                assert_equal(c.name, 'another name')
                del c
            with assert_deallocated(C, 0, 2, name='third name'):
                pass
            assert_equal(gc.isenabled(), gc_current)
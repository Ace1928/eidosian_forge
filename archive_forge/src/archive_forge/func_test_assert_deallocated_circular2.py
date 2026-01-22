import gc
from scipy._lib._gcutils import (set_gc_state, gc_state, assert_deallocated,
from numpy.testing import assert_equal
import pytest
@pytest.mark.skipif(IS_PYPY, reason='Test not meaningful on PyPy')
def test_assert_deallocated_circular2():

    class C:

        def __init__(self):
            self._circular = self
    with pytest.raises(ReferenceError):
        with assert_deallocated(C):
            pass
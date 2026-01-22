import gc
from scipy._lib._gcutils import (set_gc_state, gc_state, assert_deallocated,
from numpy.testing import assert_equal
import pytest
def test_set_gc_state():
    gc_status = gc.isenabled()
    try:
        for state in (True, False):
            gc.enable()
            set_gc_state(state)
            assert_equal(gc.isenabled(), state)
            gc.disable()
            set_gc_state(state)
            assert_equal(gc.isenabled(), state)
    finally:
        if gc_status:
            gc.enable()
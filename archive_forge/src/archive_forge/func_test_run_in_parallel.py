import inspect
import re
import warnings
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import (
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from warnings import warn
def test_run_in_parallel():
    state = []

    @run_in_parallel()
    def change_state1():
        state.append(None)
    change_state1()
    assert len(state) == 2

    @run_in_parallel(num_threads=1)
    def change_state2():
        state.append(None)
    change_state2()
    assert len(state) == 3

    @run_in_parallel(num_threads=3)
    def change_state3():
        state.append(None)
    change_state3()
    assert len(state) == 6
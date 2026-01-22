from __future__ import annotations
import warnings
import pytest
from datashader.macros import expand_varargs
import inspect
from numba import jit
def test_no_varargs_error():
    with pytest.raises(ValueError) as e:
        expand_varargs(2)(function_no_vararg)
    assert e.match('does not have a variable length positional argument')
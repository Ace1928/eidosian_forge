from __future__ import annotations
import warnings
import pytest
from datashader.macros import expand_varargs
import inspect
from numba import jit
def test_invalid_expand_number():
    with pytest.raises(ValueError) as e:
        expand_varargs(function_no_vararg)
    assert e.match('non\\-negative integer')
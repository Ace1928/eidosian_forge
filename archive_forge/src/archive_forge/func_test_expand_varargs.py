from __future__ import annotations
import warnings
import pytest
from datashader.macros import expand_varargs
import inspect
from numba import jit
def test_expand_varargs():
    assert get_args(function_with_vararg) == ['a', 'b', '*others']
    function_with_vararg_expanded = expand_varargs(2)(function_with_vararg)
    assert get_args(function_with_vararg_expanded) == ['a', 'b', '_0', '_1']
    assert function_with_vararg(1, 2, 3, 4) == function_with_vararg_expanded(1, 2, 3, 4)
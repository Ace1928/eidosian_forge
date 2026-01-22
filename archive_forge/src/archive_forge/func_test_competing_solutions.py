from __future__ import annotations
from typing import Any
from sympy.multipledispatch import dispatch
from sympy.multipledispatch.conflict import AmbiguityWarning
from sympy.testing.pytest import raises, warns
from functools import partial
def test_competing_solutions():

    @dispatch(A)
    def h(x):
        return 1

    @dispatch(C)
    def h(x):
        return 2
    assert h(D()) == 2
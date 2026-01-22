from __future__ import (absolute_import, division, print_function)
from .._sympy_Lambdify import _callback_factory
from sympy import symbols, atan
import numpy as np
import pytest
def test_callback_factory():
    args = x, y = symbols('x y')
    expr = x + atan(y)
    cb = _callback_factory(args, [expr], 'numpy', np.float64, 'C')
    inp = np.array([17, 1])
    ref = 17 + np.arctan(1)
    assert np.allclose(cb(inp), ref)
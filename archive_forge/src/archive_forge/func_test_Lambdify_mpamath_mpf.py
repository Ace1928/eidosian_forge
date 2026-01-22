from __future__ import (absolute_import, division, print_function)
from functools import reduce
from operator import add, mul
import math
import numpy as np
import pytest
from pytest import raises
from .. import Backend
def test_Lambdify_mpamath_mpf():
    import mpmath
    from mpmath import mpf
    mpmath.mp.dps = 30
    p0 = [mpf('0.7'), mpf('1.3')]
    p1 = [3]
    be = Backend('sympy')
    x, y, z = map(be.Symbol, 'xyz')
    lmb = be.Lambdify([x, y, z], [x * y * z - 1, -1 + be.exp(-y) + be.exp(-z) - 1 / x], module='mpmath')
    p = np.concatenate((p0, p1))
    lmb(p)
    lmb2 = be.Lambdify([x], [1 - x], module='mpmath', dtype=object)
    assert 9e-21 < 1 - lmb2(mpf('1e-20')) < 1.1e-20
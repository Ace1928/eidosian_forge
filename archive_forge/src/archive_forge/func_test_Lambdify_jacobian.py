from __future__ import (absolute_import, division, print_function)
from functools import reduce
from operator import add, mul
import math
import numpy as np
import pytest
from pytest import raises
from .. import Backend
@pytest.mark.parametrize('key', backends)
def test_Lambdify_jacobian(key):
    be = Backend(key)
    x = be.Symbol('x')
    y = be.Symbol('y')
    a = be.Matrix(2, 1, [x + y, y * x ** 2])
    b = be.Matrix(2, 1, [x, y])
    J = a.jacobian(b)
    lmb = be.Lambdify(b, J)
    result = lmb([3, 5])
    assert result.shape == (2, 2)
    assert np.allclose(result, [[1, 1], [2 * 3 * 5, 3 ** 2]])
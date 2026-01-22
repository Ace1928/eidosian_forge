import pytest
from numpy.f2py.symbolic import (
from . import util
def test_linear_solve(self):
    x = as_symbol('x')
    y = as_symbol('y')
    z = as_symbol('z')
    assert x.linear_solve(x) == (as_number(1), as_number(0))
    assert (x + 1).linear_solve(x) == (as_number(1), as_number(1))
    assert (2 * x).linear_solve(x) == (as_number(2), as_number(0))
    assert (2 * x + 3).linear_solve(x) == (as_number(2), as_number(3))
    assert as_number(3).linear_solve(x) == (as_number(0), as_number(3))
    assert y.linear_solve(x) == (as_number(0), y)
    assert (y * z).linear_solve(x) == (as_number(0), y * z)
    assert (x + y).linear_solve(x) == (as_number(1), y)
    assert (z * x + y).linear_solve(x) == (z, y)
    assert ((z + y) * x + y).linear_solve(x) == (z + y, y)
    assert (z * y * x + y).linear_solve(x) == (z * y, y)
    pytest.raises(RuntimeError, lambda: (x * x).linear_solve(x))
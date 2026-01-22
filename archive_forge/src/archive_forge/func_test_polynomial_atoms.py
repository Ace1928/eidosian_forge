import pytest
from numpy.f2py.symbolic import (
from . import util
def test_polynomial_atoms(self):
    x = as_symbol('x')
    y = as_symbol('y')
    n = as_number(123)
    assert x.polynomial_atoms() == {x}
    assert n.polynomial_atoms() == set()
    assert y[x].polynomial_atoms() == {y[x]}
    assert y(x).polynomial_atoms() == {y(x)}
    assert (y(x) + x).polynomial_atoms() == {y(x), x}
    assert (y(x) * x[y]).polynomial_atoms() == {y(x), x[y]}
    assert (y(x) ** x).polynomial_atoms() == {y(x)}
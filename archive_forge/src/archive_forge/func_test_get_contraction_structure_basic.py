from sympy.core import symbols, S, Pow, Function
from sympy.functions import exp
from sympy.testing.pytest import raises
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.tensor.index_methods import IndexConformanceException
from sympy.tensor.index_methods import (get_contraction_structure, get_indices)
def test_get_contraction_structure_basic():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j = (Idx('i'), Idx('j'))
    assert get_contraction_structure(x[i] * y[j]) == {None: {x[i] * y[j]}}
    assert get_contraction_structure(x[i] + y[j]) == {None: {x[i], y[j]}}
    assert get_contraction_structure(x[i] * y[i]) == {(i,): {x[i] * y[i]}}
    assert get_contraction_structure(1 + x[i] * y[i]) == {None: {S.One}, (i,): {x[i] * y[i]}}
    assert get_contraction_structure(x[i] ** y[i]) == {None: {x[i] ** y[i]}}
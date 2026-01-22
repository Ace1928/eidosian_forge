from sympy.core import symbols, S, Pow, Function
from sympy.functions import exp
from sympy.testing.pytest import raises
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.tensor.index_methods import IndexConformanceException
from sympy.tensor.index_methods import (get_contraction_structure, get_indices)
def test_get_indices_Pow():
    x = IndexedBase('x')
    y = IndexedBase('y')
    A = IndexedBase('A')
    i, j, k = (Idx('i'), Idx('j'), Idx('k'))
    assert get_indices(Pow(x[i], y[j])) == ({i, j}, {})
    assert get_indices(Pow(x[i, k], y[j, k])) == ({i, j, k}, {})
    assert get_indices(Pow(A[i, k], y[k] + A[k, j] * x[j])) == ({i, k}, {})
    assert get_indices(Pow(2, x[i])) == get_indices(exp(x[i]))
    assert get_indices(Pow(x[i], 2)) == ({i}, {})
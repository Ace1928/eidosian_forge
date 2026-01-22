from sympy.core import symbols, S, Pow, Function
from sympy.functions import exp
from sympy.testing.pytest import raises
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.tensor.index_methods import IndexConformanceException
from sympy.tensor.index_methods import (get_contraction_structure, get_indices)
def test_get_indices_Idx():
    f = Function('f')
    i, j = (Idx('i'), Idx('j'))
    assert get_indices(f(i) * j) == ({i, j}, {})
    assert get_indices(f(j, i)) == ({j, i}, {})
    assert get_indices(f(i) * i) == (set(), {})
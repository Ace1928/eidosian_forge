from sympy.core import symbols, S, Pow, Function
from sympy.functions import exp
from sympy.testing.pytest import raises
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.tensor.index_methods import IndexConformanceException
from sympy.tensor.index_methods import (get_contraction_structure, get_indices)
def test_scalar_broadcast():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j = (Idx('i'), Idx('j'))
    assert get_indices(x[i] + y[i, i]) == ({i}, {})
    assert get_indices(x[i] + y[j, j]) == ({i}, {})
from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag
def test_no_metric_symmetry():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L', metric_symmetry=0)
    d0, d1, d2, d3 = tensor_indices('d:4', Lorentz)
    A = TensorHead('A', [Lorentz] * 2, TensorSymmetry.no_symmetry(2))
    t = A(d1, -d0) * A(d0, -d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_0)'
    t = A(d1, -d2) * A(d0, -d3) * A(d2, -d1) * A(d3, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_0)*A(L_2, -L_3)*A(L_3, -L_2)'
    t = A(d0, -d1) * A(d1, -d2) * A(d2, -d3) * A(d3, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_2)*A(L_2, -L_3)*A(L_3, -L_0)'
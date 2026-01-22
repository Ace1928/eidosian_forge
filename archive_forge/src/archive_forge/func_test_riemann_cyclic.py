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
def test_riemann_cyclic():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    i, j, k, l, m, n, p, q = tensor_indices('i,j,k,l,m,n,p,q', Lorentz)
    R = TensorHead('R', [Lorentz] * 4, TensorSymmetry.riemann())
    t = R(i, j, k, l) + R(i, l, j, k) + R(i, k, l, j) - R(i, j, l, k) - R(i, l, k, j) - R(i, k, j, l)
    t2 = t * R(-i, -j, -k, -l)
    t3 = riemann_cyclic(t2)
    assert t3 == 0
    t = R(i, j, k, l) * (R(-i, -j, -k, -l) - 2 * R(-i, -k, -j, -l))
    t1 = riemann_cyclic(t)
    assert t1 == 0
    t = R(i, j, k, l)
    t1 = riemann_cyclic(t)
    assert t1 == Rational(-1, 3) * R(i, l, j, k) + Rational(1, 3) * R(i, k, j, l) + Rational(2, 3) * R(i, j, k, l)
    t = R(i, j, k, l) * R(-k, -l, m, n) * (R(-m, -n, -i, -j) + 2 * R(-m, -j, -n, -i))
    t1 = riemann_cyclic(t)
    assert t1 == 0
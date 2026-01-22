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
def test_issue_11020_TensAdd_data():
    with warns_deprecated_sympy():
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='i', dim=2)
        Lorentz.data = [-1, 1]
        a, b, c, d = tensor_indices('a, b, c, d', Lorentz)
        i0, i1 = tensor_indices('i_0:2', Lorentz)
        g = TensorHead('g', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
        g.data = Lorentz.data
        u = TensorHead('u', [Lorentz])
        u.data = [1, 0]
        add_1 = g(b, c) * g(d, i0) * u(-i0) - g(b, c) * u(d)
        assert add_1.data == Array.zeros(2, 2, 2)
        add_2 = g(b, c) * g(a, i0) * u(-i0) - g(b, c) * u(a)
        assert add_2.data == Array.zeros(2, 2, 2)
        perp = u(a) * u(b) + g(a, b)
        mul_1 = u(-a) * perp(a, b)
        assert mul_1.data == Array([0, 0])
        mul_2 = u(-c) * perp(c, a) * perp(d, b)
        assert mul_2.data == Array.zeros(2, 2, 2)
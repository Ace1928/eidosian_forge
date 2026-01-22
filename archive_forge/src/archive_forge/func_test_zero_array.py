import random
from sympy import tensordiagonal, eye, KroneckerDelta, Array
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.combinatorics import Permutation
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, ArraySymbol, ArrayElement, \
from sympy.testing.pytest import raises
def test_zero_array():
    assert ZeroArray() == 0
    assert ZeroArray().is_Integer
    za = ZeroArray(3, 2, 4)
    assert za.shape == (3, 2, 4)
    za_e = za.as_explicit()
    assert za_e.shape == (3, 2, 4)
    m, n, k = symbols('m n k')
    za = ZeroArray(m, n, k, 2)
    assert za.shape == (m, n, k, 2)
    raises(ValueError, lambda: za.as_explicit())
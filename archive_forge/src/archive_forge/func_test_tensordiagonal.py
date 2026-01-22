import itertools
import random
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.testing.pytest import raises
from sympy.core.function import diff
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.tensor.array import Array, ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableSparseNDimArray
from sympy.tensor.array.arrayop import tensorproduct, tensorcontraction, derive_by_array, permutedims, Flatten, \
def test_tensordiagonal():
    from sympy.matrices.dense import eye
    expr = Array(range(9)).reshape(3, 3)
    raises(ValueError, lambda: tensordiagonal(expr, [0], [1]))
    raises(ValueError, lambda: tensordiagonal(expr, [0, 0]))
    assert tensordiagonal(eye(3), [0, 1]) == Array([1, 1, 1])
    assert tensordiagonal(expr, [0, 1]) == Array([0, 4, 8])
    x, y, z = symbols('x y z')
    expr2 = tensorproduct([x, y, z], expr)
    assert tensordiagonal(expr2, [1, 2]) == Array([[0, 4 * x, 8 * x], [0, 4 * y, 8 * y], [0, 4 * z, 8 * z]])
    assert tensordiagonal(expr2, [0, 1]) == Array([[0, 3 * y, 6 * z], [x, 4 * y, 7 * z], [2 * x, 5 * y, 8 * z]])
    assert tensordiagonal(expr2, [0, 1, 2]) == Array([0, 4 * y, 8 * z])
    a, b, c, X, Y, Z = symbols('a b c X Y Z')
    expr3 = tensorproduct([x, y, z], [1, 2, 3], [a, b, c], [X, Y, Z])
    assert tensordiagonal(expr3, [0, 1, 2, 3]) == Array([x * a * X, 2 * y * b * Y, 3 * z * c * Z])
    assert tensordiagonal(expr3, [0, 1], [2, 3]) == tensorproduct([x, 2 * y, 3 * z], [a * X, b * Y, c * Z])
    assert tensordiagonal(tensordiagonal(expr3, [2, 3]), [0, 1]) == tensorproduct([a * X, b * Y, c * Z], [x, 2 * y, 3 * z])
    raises(ValueError, lambda: tensordiagonal([[1, 2, 3], [4, 5, 6]], [0, 1]))
    raises(ValueError, lambda: tensordiagonal(expr3.reshape(3, 3, 9), [1, 2]))
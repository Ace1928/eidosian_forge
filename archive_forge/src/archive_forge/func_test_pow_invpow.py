from functools import reduce
import itertools
from operator import add
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import Inverse, MatAdd, MatMul, Transpose
from sympy.polys.rootoftools import CRootOf
from sympy.series.order import O
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import signsimp
from sympy.tensor.indexed import (Idx, IndexedBase)
from sympy.core.function import count_ops
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.functions.special.hyper import meijerg
from sympy.simplify import cse_main, cse_opts
from sympy.utilities.iterables import subsets
from sympy.testing.pytest import XFAIL, raises
from sympy.matrices import (MutableDenseMatrix, MutableSparseMatrix,
from sympy.matrices.expressions import MatrixSymbol
def test_pow_invpow():
    assert cse(1 / x ** 2 + x ** 2) == ([(x0, x ** 2)], [x0 + 1 / x0])
    assert cse(x ** 2 + (1 + 1 / x ** 2) / x ** 2) == ([(x0, x ** 2), (x1, 1 / x0)], [x0 + x1 * (x1 + 1)])
    assert cse(1 / x ** 2 + (1 + 1 / x ** 2) * x ** 2) == ([(x0, x ** 2), (x1, 1 / x0)], [x0 * (x1 + 1) + x1])
    assert cse(cos(1 / x ** 2) + sin(1 / x ** 2)) == ([(x0, x ** (-2))], [sin(x0) + cos(x0)])
    assert cse(cos(x ** 2) + sin(x ** 2)) == ([(x0, x ** 2)], [sin(x0) + cos(x0)])
    assert cse(y / (2 + x ** 2) + z / x ** 2 / y) == ([(x0, x ** 2)], [y / (x0 + 2) + z / (x0 * y)])
    assert cse(exp(x ** 2) + x ** 2 * cos(1 / x ** 2)) == ([(x0, x ** 2)], [x0 * cos(1 / x0) + exp(x0)])
    assert cse((1 + 1 / x ** 2) / x ** 2) == ([(x0, x ** (-2))], [x0 * (x0 + 1)])
    assert cse(x ** (2 * y) + x ** (-2 * y)) == ([(x0, x ** (2 * y))], [x0 + 1 / x0])
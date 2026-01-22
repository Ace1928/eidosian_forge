from math import prod
from sympy.concrete.expr_with_intlimits import ReorderError
from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import (Sum, summation, telescopic,
from sympy.core.function import (Derivative, Function)
from sympy.core import (Catalan, EulerGamma)
from sympy.core.facts import InconsistentAssumptions
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (sinh, tanh)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Or
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices import (Matrix, SparseMatrix,
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.abc import a, b, c, d, k, m, x, y, z
def test_expand_with_assumptions():
    M = Symbol('M', integer=True, positive=True)
    x = Symbol('x', positive=True)
    m = Symbol('m', nonnegative=True)
    assert log(Product(x ** m, (m, 0, M))).expand() == Sum(m * log(x), (m, 0, M))
    assert log(Product(exp(x ** m), (m, 0, M))).expand() == Sum(x ** m, (m, 0, M))
    assert log(Product(x ** m, (m, 0, M))).rewrite(Sum).expand() == Sum(m * log(x), (m, 0, M))
    assert log(Product(exp(x ** m), (m, 0, M))).rewrite(Sum).expand() == Sum(x ** m, (m, 0, M))
    n = Symbol('n', nonnegative=True)
    i, j = symbols('i,j', positive=True, integer=True)
    x, y = symbols('x,y', positive=True)
    assert log(Product(x ** i * y ** j, (i, 1, n), (j, 1, m))).expand() == Sum(i * log(x) + j * log(y), (i, 1, n), (j, 1, m))
    m = Symbol('m', nonnegative=True, integer=True)
    s = Sum(x ** m, (m, 0, M))
    s_as_product = s.rewrite(Product)
    assert s_as_product.has(Product)
    assert s_as_product == log(Product(exp(x ** m), (m, 0, M)))
    assert s_as_product.expand() == s
    s5 = s.subs(M, 5)
    s5_as_product = s5.rewrite(Product)
    assert s5_as_product.has(Product)
    assert s5_as_product.doit().expand() == s5.doit()
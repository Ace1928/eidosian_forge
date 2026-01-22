from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.elementary.complexes import polar_lift
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besselk
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices.dense import eye
from sympy.matrices.expressions.determinant import Determinant
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Interval, ProductSet)
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.numbers import comp
from sympy.integrals.integrals import integrate
from sympy.matrices import Matrix, MatrixSymbol
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.stats import density, median, marginal_distribution, Normal, Laplace, E, sample
from sympy.stats.joint_rv_types import (JointRV, MultivariateNormalDistribution,
from sympy.testing.pytest import raises, XFAIL, skip, slow
from sympy.external import import_module
from sympy.abc import x, y
def test_NegativeMultinomial():
    k0, x1, x2, x3, x4 = symbols('k0, x1, x2, x3, x4', nonnegative=True, integer=True)
    p1, p2, p3, p4 = symbols('p1, p2, p3, p4', positive=True)
    p1_f = symbols('p1_f', negative=True)
    N = NegativeMultinomial('N', 4, [p1, p2, p3, p4])
    C = NegativeMultinomial('C', 4, 0.1, 0.2, 0.3)
    g = gamma
    f = factorial
    assert simplify(density(N)(x1, x2, x3, x4) - p1 ** x1 * p2 ** x2 * p3 ** x3 * p4 ** x4 * (-p1 - p2 - p3 - p4 + 1) ** 4 * g(x1 + x2 + x3 + x4 + 4) / (6 * f(x1) * f(x2) * f(x3) * f(x4))) is S.Zero
    assert comp(marginal_distribution(C, C[0])(1).evalf(), 0.33, 0.01)
    raises(ValueError, lambda: NegativeMultinomial('b1', 5, [p1, p2, p3, p1_f]))
    raises(ValueError, lambda: NegativeMultinomial('b2', k0, 0.5, 0.4, 0.3, 0.4))
    assert N.pspace.distribution.set == ProductSet(Range(0, oo, 1), Range(0, oo, 1), Range(0, oo, 1), Range(0, oo, 1))
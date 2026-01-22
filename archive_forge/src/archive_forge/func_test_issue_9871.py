from itertools import product
import math
import inspect
import mpmath
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, Lambda, diff)
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, cos, cot, sin,
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely)
from sympy.functions.special.beta_functions import (beta, betainc, betainc_regularized)
from sympy.functions.special.delta_functions import (Heaviside)
from sympy.functions.special.error_functions import (Ei, erf, erfc, fresnelc, fresnels, Si, Ci)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, polygamma)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, false, ITE, Not, Or, true)
from sympy.matrices.expressions.dotproduct import DotProduct
from sympy.tensor.array import derive_by_array, Array
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.lambdify import lambdify
from sympy.core.expr import UnevaluatedExpr
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, log10, hypot
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.functions.elementary.complexes import re, im, arg
from sympy.functions.special.polynomials import \
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.printing.numpy import NumPyPrinter
from sympy.utilities.lambdify import implemented_function, lambdastr
from sympy.testing.pytest import skip
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.exceptions import ignore_warnings
from sympy.external import import_module
from sympy.functions.special.gamma_functions import uppergamma, lowergamma
import sympy
def test_issue_9871():
    if not numexpr:
        skip('numexpr not installed.')
    if not numpy:
        skip('numpy not installed.')
    r = sqrt(x ** 2 + y ** 2)
    expr = diff(1 / r, x)
    xn = yn = numpy.linspace(1, 10, 16)
    fv_exact = -numpy.sqrt(2.0) ** (-3) * xn ** (-2)
    fv_numpy = lambdify((x, y), expr, modules='numpy')(xn, yn)
    fv_numexpr = lambdify((x, y), expr, modules='numexpr')(xn, yn)
    numpy.testing.assert_allclose(fv_numpy, fv_exact, rtol=1e-10)
    numpy.testing.assert_allclose(fv_numexpr, fv_exact, rtol=1e-10)
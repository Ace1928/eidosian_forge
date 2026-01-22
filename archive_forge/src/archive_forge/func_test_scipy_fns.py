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
def test_scipy_fns():
    if not scipy:
        skip('scipy not installed')
    single_arg_sympy_fns = [Ei, erf, erfc, factorial, gamma, loggamma, digamma, Si, Ci]
    single_arg_scipy_fns = [scipy.special.expi, scipy.special.erf, scipy.special.erfc, scipy.special.factorial, scipy.special.gamma, scipy.special.gammaln, scipy.special.psi, scipy.special.sici, scipy.special.sici]
    numpy.random.seed(0)
    for sympy_fn, scipy_fn in zip(single_arg_sympy_fns, single_arg_scipy_fns):
        f = lambdify(x, sympy_fn(x), modules='scipy')
        for i in range(20):
            tv = numpy.random.uniform(-10, 10) + 1j * numpy.random.uniform(-5, 5)
            if sympy_fn == factorial:
                tv = numpy.abs(tv)
            if sympy_fn == loggamma:
                tv = numpy.abs(tv)
            if sympy_fn == digamma:
                tv = numpy.real(tv)
            sympy_result = sympy_fn(tv).evalf()
            scipy_result = scipy_fn(tv)
            if sympy_fn == Si:
                scipy_result = scipy_fn(tv)[0]
            if sympy_fn == Ci:
                scipy_result = scipy_fn(tv)[1]
            assert abs(f(tv) - sympy_result) < 1e-13 * (1 + abs(sympy_result))
            assert abs(f(tv) - scipy_result) < 1e-13 * (1 + abs(sympy_result))
    double_arg_sympy_fns = [RisingFactorial, besselj, bessely, besseli, besselk, polygamma]
    double_arg_scipy_fns = [scipy.special.poch, scipy.special.jv, scipy.special.yv, scipy.special.iv, scipy.special.kv, scipy.special.polygamma]
    for sympy_fn, scipy_fn in zip(double_arg_sympy_fns, double_arg_scipy_fns):
        f = lambdify((x, y), sympy_fn(x, y), modules='scipy')
        for i in range(20):
            tv1 = numpy.random.uniform(-10, 10)
            tv2 = numpy.random.uniform(-10, 10) + 1j * numpy.random.uniform(-5, 5)
            if sympy_fn in (RisingFactorial, polygamma):
                tv2 = numpy.real(tv2)
            if sympy_fn == polygamma:
                tv1 = abs(int(tv1))
            sympy_result = sympy_fn(tv1, tv2).evalf()
            assert abs(f(tv1, tv2) - sympy_result) < 1e-13 * (1 + abs(sympy_result))
            assert abs(f(tv1, tv2) - scipy_fn(tv1, tv2)) < 1e-13 * (1 + abs(sympy_result))
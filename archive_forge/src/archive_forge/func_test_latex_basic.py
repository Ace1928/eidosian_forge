from sympy import MatAdd, MatMul, Array
from sympy.algebras.quaternion import Quaternion
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.combinatorics.permutations import Cycle, Permutation, AppliedPermutation
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple, Dict
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import (Derivative, Function, Lambda, Subs, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (AlgebraicNumber, Float, I, Integer, Rational, oo, pi)
from sympy.core.parameters import evaluate
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.combinatorial.factorials import (FallingFactorial, RisingFactorial, binomial, factorial, factorial2, subfactorial)
from sympy.functions.combinatorial.numbers import bernoulli, bell, catalan, euler, genocchi, lucas, fibonacci, tribonacci
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, polar_lift, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, coth)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (Max, Min, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acsc, asin, cos, cot, sin, tan)
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f, elliptic_k, elliptic_pi)
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, expint)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.functions.special.mathieu_functions import (mathieuc, mathieucprime, mathieus, mathieusprime)
from sympy.functions.special.polynomials import (assoc_laguerre, assoc_legendre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.spherical_harmonics import (Ynm, Znm)
from sympy.functions.special.tensor_functions import (KroneckerDelta, LeviCivita)
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, stieltjes, zeta)
from sympy.integrals.integrals import Integral
from sympy.integrals.transforms import (CosineTransform, FourierTransform, InverseCosineTransform, InverseFourierTransform, InverseLaplaceTransform, InverseMellinTransform, InverseSineTransform, LaplaceTransform, MellinTransform, SineTransform)
from sympy.logic import Implies
from sympy.logic.boolalg import (And, Or, Xor, Equivalent, false, Not, true)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.kronecker import KroneckerProduct
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.permutation import PermutationMatrix
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.physics.control.lti import TransferFunction, Series, Parallel, Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback
from sympy.ntheory.factor_ import (divisor_sigma, primenu, primeomega, reduced_totient, totient, udivisor_sigma)
from sympy.physics.quantum import Commutator, Operator
from sympy.physics.quantum.trace import Tr
from sympy.physics.units import meter, gibibyte, gram, microgram, second, milli, micro
from sympy.polys.domains.integerring import ZZ
from sympy.polys.fields import field
from sympy.polys.polytools import Poly
from sympy.polys.rings import ring
from sympy.polys.rootoftools import (RootSum, rootof)
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import (SeqAdd, SeqFormula, SeqMul, SeqPer)
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ComplexRegion, ImageSet, Range)
from sympy.sets.ordinals import Ordinal, OrdinalOmega, OmegaPower
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import (FiniteSet, Interval, Union, Intersection, Complement, SymmetricDifference, ProductSet)
from sympy.sets.setexpr import SetExpr
from sympy.stats.crv_types import Normal
from sympy.stats.symbolic_probability import (Covariance, Expectation,
from sympy.tensor.array import (ImmutableDenseNDimArray,
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElement
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)
from sympy.tensor.toperators import PartialDerivative
from sympy.vector import CoordSys3D, Cross, Curl, Dot, Divergence, Gradient, Laplacian
from sympy.testing.pytest import (XFAIL, raises, _both_exp_pow,
from sympy.printing.latex import (latex, translate, greek_letters_set,
import sympy as sym
from sympy.abc import mu, tau
def test_latex_basic():
    assert latex(1 + x) == 'x + 1'
    assert latex(x ** 2) == 'x^{2}'
    assert latex(x ** (1 + x)) == 'x^{x + 1}'
    assert latex(x ** 3 + x + 1 + x ** 2) == 'x^{3} + x^{2} + x + 1'
    assert latex(2 * x * y) == '2 x y'
    assert latex(2 * x * y, mul_symbol='dot') == '2 \\cdot x \\cdot y'
    assert latex(3 * x ** 2 * y, mul_symbol='\\,') == '3\\,x^{2}\\,y'
    assert latex(1.5 * 3 ** x, mul_symbol='\\,') == '1.5 \\cdot 3^{x}'
    assert latex(x ** S.Half ** 5) == '\\sqrt[32]{x}'
    assert latex(Mul(S.Half, x ** 2, -5, evaluate=False)) == '\\frac{1}{2} x^{2} \\left(-5\\right)'
    assert latex(Mul(S.Half, x ** 2, 5, evaluate=False)) == '\\frac{1}{2} x^{2} \\cdot 5'
    assert latex(Mul(-5, -5, evaluate=False)) == '\\left(-5\\right) \\left(-5\\right)'
    assert latex(Mul(5, -5, evaluate=False)) == '5 \\left(-5\\right)'
    assert latex(Mul(S.Half, -5, S.Half, evaluate=False)) == '\\frac{1}{2} \\left(-5\\right) \\frac{1}{2}'
    assert latex(Mul(5, I, 5, evaluate=False)) == '5 i 5'
    assert latex(Mul(5, I, -5, evaluate=False)) == '5 i \\left(-5\\right)'
    assert latex(Mul(0, 1, evaluate=False)) == '0 \\cdot 1'
    assert latex(Mul(1, 0, evaluate=False)) == '1 \\cdot 0'
    assert latex(Mul(1, 1, evaluate=False)) == '1 \\cdot 1'
    assert latex(Mul(-1, 1, evaluate=False)) == '\\left(-1\\right) 1'
    assert latex(Mul(1, 1, 1, evaluate=False)) == '1 \\cdot 1 \\cdot 1'
    assert latex(Mul(1, 2, evaluate=False)) == '1 \\cdot 2'
    assert latex(Mul(1, S.Half, evaluate=False)) == '1 \\cdot \\frac{1}{2}'
    assert latex(Mul(1, 1, S.Half, evaluate=False)) == '1 \\cdot 1 \\cdot \\frac{1}{2}'
    assert latex(Mul(1, 1, 2, 3, x, evaluate=False)) == '1 \\cdot 1 \\cdot 2 \\cdot 3 x'
    assert latex(Mul(1, -1, evaluate=False)) == '1 \\left(-1\\right)'
    assert latex(Mul(4, 3, 2, 1, 0, y, x, evaluate=False)) == '4 \\cdot 3 \\cdot 2 \\cdot 1 \\cdot 0 y x'
    assert latex(Mul(4, 3, 2, 1 + z, 0, y, x, evaluate=False)) == '4 \\cdot 3 \\cdot 2 \\left(z + 1\\right) 0 y x'
    assert latex(Mul(Rational(2, 3), Rational(5, 7), evaluate=False)) == '\\frac{2}{3} \\cdot \\frac{5}{7}'
    assert latex(1 / x) == '\\frac{1}{x}'
    assert latex(1 / x, fold_short_frac=True) == '1 / x'
    assert latex(-S(3) / 2) == '- \\frac{3}{2}'
    assert latex(-S(3) / 2, fold_short_frac=True) == '- 3 / 2'
    assert latex(1 / x ** 2) == '\\frac{1}{x^{2}}'
    assert latex(1 / (x + y) / 2) == '\\frac{1}{2 \\left(x + y\\right)}'
    assert latex(x / 2) == '\\frac{x}{2}'
    assert latex(x / 2, fold_short_frac=True) == 'x / 2'
    assert latex((x + y) / (2 * x)) == '\\frac{x + y}{2 x}'
    assert latex((x + y) / (2 * x), fold_short_frac=True) == '\\left(x + y\\right) / 2 x'
    assert latex((x + y) / (2 * x), long_frac_ratio=0) == '\\frac{1}{2 x} \\left(x + y\\right)'
    assert latex((x + y) / x) == '\\frac{x + y}{x}'
    assert latex((x + y) / x, long_frac_ratio=3) == '\\frac{x + y}{x}'
    assert latex(2 * sqrt(2) * x / 3) == '\\frac{2 \\sqrt{2} x}{3}'
    assert latex(2 * sqrt(2) * x / 3, long_frac_ratio=2) == '\\frac{2 x}{3} \\sqrt{2}'
    assert latex(binomial(x, y)) == '{\\binom{x}{y}}'
    x_star = Symbol('x^*')
    f = Function('f')
    assert latex(x_star ** 2) == '\\left(x^{*}\\right)^{2}'
    assert latex(x_star ** 2, parenthesize_super=False) == '{x^{*}}^{2}'
    assert latex(Derivative(f(x_star), x_star, 2)) == '\\frac{d^{2}}{d \\left(x^{*}\\right)^{2}} f{\\left(x^{*} \\right)}'
    assert latex(Derivative(f(x_star), x_star, 2), parenthesize_super=False) == '\\frac{d^{2}}{d {x^{*}}^{2}} f{\\left(x^{*} \\right)}'
    assert latex(2 * Integral(x, x) / 3) == '\\frac{2 \\int x\\, dx}{3}'
    assert latex(2 * Integral(x, x) / 3, fold_short_frac=True) == '\\left(2 \\int x\\, dx\\right) / 3'
    assert latex(sqrt(x)) == '\\sqrt{x}'
    assert latex(x ** Rational(1, 3)) == '\\sqrt[3]{x}'
    assert latex(x ** Rational(1, 3), root_notation=False) == 'x^{\\frac{1}{3}}'
    assert latex(sqrt(x) ** 3) == 'x^{\\frac{3}{2}}'
    assert latex(sqrt(x), itex=True) == '\\sqrt{x}'
    assert latex(x ** Rational(1, 3), itex=True) == '\\root{3}{x}'
    assert latex(sqrt(x) ** 3, itex=True) == 'x^{\\frac{3}{2}}'
    assert latex(x ** Rational(3, 4)) == 'x^{\\frac{3}{4}}'
    assert latex(x ** Rational(3, 4), fold_frac_powers=True) == 'x^{3/4}'
    assert latex((x + 1) ** Rational(3, 4)) == '\\left(x + 1\\right)^{\\frac{3}{4}}'
    assert latex((x + 1) ** Rational(3, 4), fold_frac_powers=True) == '\\left(x + 1\\right)^{3/4}'
    assert latex(AlgebraicNumber(sqrt(2))) == '\\sqrt{2}'
    assert latex(AlgebraicNumber(sqrt(2), [3, -7])) == '-7 + 3 \\sqrt{2}'
    assert latex(AlgebraicNumber(sqrt(2), alias='alpha')) == '\\alpha'
    assert latex(AlgebraicNumber(sqrt(2), [3, -7], alias='alpha')) == '3 \\alpha - 7'
    assert latex(AlgebraicNumber(2 ** (S(1) / 3), [1, 3, -7], alias='beta')) == '\\beta^{2} + 3 \\beta - 7'
    k = ZZ.cyclotomic_field(5)
    assert latex(k.ext.field_element([1, 2, 3, 4])) == '\\zeta^{3} + 2 \\zeta^{2} + 3 \\zeta + 4'
    assert latex(k.ext.field_element([1, 2, 3, 4]), order='old') == '4 + 3 \\zeta + 2 \\zeta^{2} + \\zeta^{3}'
    assert latex(k.primes_above(19)[0]) == '\\left(19, \\zeta^{2} + 5 \\zeta + 1\\right)'
    assert latex(k.primes_above(19)[0], order='old') == '\\left(19, 1 + 5 \\zeta + \\zeta^{2}\\right)'
    assert latex(k.primes_above(7)[0]) == '\\left(7\\right)'
    assert latex(1.5e+20 * x) == '1.5 \\cdot 10^{20} x'
    assert latex(1.5e+20 * x, mul_symbol='dot') == '1.5 \\cdot 10^{20} \\cdot x'
    assert latex(1.5e+20 * x, mul_symbol='times') == '1.5 \\times 10^{20} \\times x'
    assert latex(1 / sin(x)) == '\\frac{1}{\\sin{\\left(x \\right)}}'
    assert latex(sin(x) ** (-1)) == '\\frac{1}{\\sin{\\left(x \\right)}}'
    assert latex(sin(x) ** Rational(3, 2)) == '\\sin^{\\frac{3}{2}}{\\left(x \\right)}'
    assert latex(sin(x) ** Rational(3, 2), fold_frac_powers=True) == '\\sin^{3/2}{\\left(x \\right)}'
    assert latex(~x) == '\\neg x'
    assert latex(x & y) == 'x \\wedge y'
    assert latex(x & y & z) == 'x \\wedge y \\wedge z'
    assert latex(x | y) == 'x \\vee y'
    assert latex(x | y | z) == 'x \\vee y \\vee z'
    assert latex(x & y | z) == 'z \\vee \\left(x \\wedge y\\right)'
    assert latex(Implies(x, y)) == 'x \\Rightarrow y'
    assert latex(~(x >> ~y)) == 'x \\not\\Rightarrow \\neg y'
    assert latex(Implies(Or(x, y), z)) == '\\left(x \\vee y\\right) \\Rightarrow z'
    assert latex(Implies(z, Or(x, y))) == 'z \\Rightarrow \\left(x \\vee y\\right)'
    assert latex(~(x & y)) == '\\neg \\left(x \\wedge y\\right)'
    assert latex(~x, symbol_names={x: 'x_i'}) == '\\neg x_i'
    assert latex(x & y, symbol_names={x: 'x_i', y: 'y_i'}) == 'x_i \\wedge y_i'
    assert latex(x & y & z, symbol_names={x: 'x_i', y: 'y_i', z: 'z_i'}) == 'x_i \\wedge y_i \\wedge z_i'
    assert latex(x | y, symbol_names={x: 'x_i', y: 'y_i'}) == 'x_i \\vee y_i'
    assert latex(x | y | z, symbol_names={x: 'x_i', y: 'y_i', z: 'z_i'}) == 'x_i \\vee y_i \\vee z_i'
    assert latex(x & y | z, symbol_names={x: 'x_i', y: 'y_i', z: 'z_i'}) == 'z_i \\vee \\left(x_i \\wedge y_i\\right)'
    assert latex(Implies(x, y), symbol_names={x: 'x_i', y: 'y_i'}) == 'x_i \\Rightarrow y_i'
    assert latex(Pow(Rational(1, 3), -1, evaluate=False)) == '\\frac{1}{\\frac{1}{3}}'
    assert latex(Pow(Rational(1, 3), -2, evaluate=False)) == '\\frac{1}{(\\frac{1}{3})^{2}}'
    assert latex(Pow(Integer(1) / 100, -1, evaluate=False)) == '\\frac{1}{\\frac{1}{100}}'
    p = Symbol('p', positive=True)
    assert latex(exp(-p) * log(p)) == 'e^{- p} \\log{\\left(p \\right)}'
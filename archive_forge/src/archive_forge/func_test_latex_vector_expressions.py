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
def test_latex_vector_expressions():
    A = CoordSys3D('A')
    assert latex(Cross(A.i, A.j * A.x * 3 + A.k)) == '\\mathbf{\\hat{i}_{A}} \\times \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}} + \\mathbf{\\hat{k}_{A}}\\right)'
    assert latex(Cross(A.i, A.j)) == '\\mathbf{\\hat{i}_{A}} \\times \\mathbf{\\hat{j}_{A}}'
    assert latex(x * Cross(A.i, A.j)) == 'x \\left(\\mathbf{\\hat{i}_{A}} \\times \\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(Cross(x * A.i, A.j)) == '- \\mathbf{\\hat{j}_{A}} \\times \\left(\\left(x\\right)\\mathbf{\\hat{i}_{A}}\\right)'
    assert latex(Curl(3 * A.x * A.j)) == '\\nabla\\times \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(Curl(3 * A.x * A.j + A.i)) == '\\nabla\\times \\left(\\mathbf{\\hat{i}_{A}} + \\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(Curl(3 * x * A.x * A.j)) == '\\nabla\\times \\left(\\left(3 \\mathbf{{x}_{A}} x\\right)\\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(x * Curl(3 * A.x * A.j)) == 'x \\left(\\nabla\\times \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)\\right)'
    assert latex(Divergence(3 * A.x * A.j + A.i)) == '\\nabla\\cdot \\left(\\mathbf{\\hat{i}_{A}} + \\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(Divergence(3 * A.x * A.j)) == '\\nabla\\cdot \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(x * Divergence(3 * A.x * A.j)) == 'x \\left(\\nabla\\cdot \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)\\right)'
    assert latex(Dot(A.i, A.j * A.x * 3 + A.k)) == '\\mathbf{\\hat{i}_{A}} \\cdot \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}} + \\mathbf{\\hat{k}_{A}}\\right)'
    assert latex(Dot(A.i, A.j)) == '\\mathbf{\\hat{i}_{A}} \\cdot \\mathbf{\\hat{j}_{A}}'
    assert latex(Dot(x * A.i, A.j)) == '\\mathbf{\\hat{j}_{A}} \\cdot \\left(\\left(x\\right)\\mathbf{\\hat{i}_{A}}\\right)'
    assert latex(x * Dot(A.i, A.j)) == 'x \\left(\\mathbf{\\hat{i}_{A}} \\cdot \\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(Gradient(A.x)) == '\\nabla \\mathbf{{x}_{A}}'
    assert latex(Gradient(A.x + 3 * A.y)) == '\\nabla \\left(\\mathbf{{x}_{A}} + 3 \\mathbf{{y}_{A}}\\right)'
    assert latex(x * Gradient(A.x)) == 'x \\left(\\nabla \\mathbf{{x}_{A}}\\right)'
    assert latex(Gradient(x * A.x)) == '\\nabla \\left(\\mathbf{{x}_{A}} x\\right)'
    assert latex(Laplacian(A.x)) == '\\Delta \\mathbf{{x}_{A}}'
    assert latex(Laplacian(A.x + 3 * A.y)) == '\\Delta \\left(\\mathbf{{x}_{A}} + 3 \\mathbf{{y}_{A}}\\right)'
    assert latex(x * Laplacian(A.x)) == 'x \\left(\\Delta \\mathbf{{x}_{A}}\\right)'
    assert latex(Laplacian(x * A.x)) == '\\Delta \\left(\\mathbf{{x}_{A}} x\\right)'
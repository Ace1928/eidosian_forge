from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import (Derivative, Function, Lambda, Subs)
from sympy.core.mul import Mul
from sympy.core import (EulerGamma, GoldenRatio, Catalan)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import LambertW
from sympy.functions.special.bessel import (airyai, airyaiprime, airybi, airybiprime)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.error_functions import (fresnelc, fresnels)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import dirichlet_eta
from sympy.geometry.line import (Ray, Segment)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Equivalent, ITE, Implies, Nand, Nor, Not, Or, Xor)
from sympy.matrices.dense import (Matrix, diag)
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.trace import Trace
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.realfield import RR
from sympy.polys.orderings import (grlex, ilex)
from sympy.polys.polytools import groebner
from sympy.polys.rootoftools import (RootSum, rootof)
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.series.limits import Limit
from sympy.series.order import O
from sympy.series.sequences import (SeqAdd, SeqFormula, SeqMul, SeqPer)
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Complement, FiniteSet, Intersection, Interval, Union)
from sympy.codegen.ast import (Assignment, AddAugmentedAssignment,
from sympy.core.expr import UnevaluatedExpr
from sympy.physics.quantum.trace import Tr
from sympy.functions import (Abs, Chi, Ci, Ei, KroneckerDelta,
from sympy.matrices import (Adjoint, Inverse, MatrixSymbol, Transpose,
from sympy.matrices.expressions import hadamard_power
from sympy.physics import mechanics
from sympy.physics.control.lti import (TransferFunction, Feedback, TransferFunctionMatrix,
from sympy.physics.units import joule, degree
from sympy.printing.pretty import pprint, pretty as xpretty
from sympy.printing.pretty.pretty_symbology import center_accent, is_combining
from sympy.sets.conditionset import ConditionSet
from sympy.sets import ImageSet, ProductSet
from sympy.sets.setexpr import SetExpr
from sympy.stats.crv_types import Normal
from sympy.stats.symbolic_probability import (Covariance, Expectation,
from sympy.tensor.array import (ImmutableDenseNDimArray, ImmutableSparseNDimArray,
from sympy.tensor.functions import TensorProduct
from sympy.tensor.tensor import (TensorIndexType, tensor_indices, TensorHead,
from sympy.testing.pytest import raises, _both_exp_pow, warns_deprecated_sympy
from sympy.vector import CoordSys3D, Gradient, Curl, Divergence, Dot, Cross, Laplacian
import sympy as sym
def test_pretty_integrals():
    expr = Integral(log(x), x)
    ascii_str = '  /         \n |          \n | log(x) dx\n |          \n/           '
    ucode_str = '⌠          \n⎮ log(x) dx\n⌡          '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Integral(x ** 2, x)
    ascii_str = '  /     \n |      \n |  2   \n | x  dx\n |      \n/       '
    ucode_str = '⌠      \n⎮  2   \n⎮ x  dx\n⌡      '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Integral(sin(x) ** 2 / tan(x) ** 2)
    ascii_str = '  /          \n |           \n |    2      \n | sin (x)   \n | ------- dx\n |    2      \n | tan (x)   \n |           \n/            '
    ucode_str = '⌠           \n⎮    2      \n⎮ sin (x)   \n⎮ ─────── dx\n⎮    2      \n⎮ tan (x)   \n⌡           '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Integral(x ** 2 ** x, x)
    ascii_str = '  /        \n |         \n |  / x\\   \n |  \\2 /   \n | x     dx\n |         \n/          '
    ucode_str = '⌠         \n⎮  ⎛ x⎞   \n⎮  ⎝2 ⎠   \n⎮ x     dx\n⌡         '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Integral(x ** 2, (x, 1, 2))
    ascii_str = '  2      \n  /      \n |       \n |   2   \n |  x  dx\n |       \n/        \n1        '
    ucode_str = '2      \n⌠      \n⎮  2   \n⎮ x  dx\n⌡      \n1      '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Integral(x ** 2, (x, Rational(1, 2), 10))
    ascii_str = ' 10      \n  /      \n |       \n |   2   \n |  x  dx\n |       \n/        \n1/2      '
    ucode_str = ' 10      \n ⌠       \n ⎮   2   \n ⎮  x  dx\n ⌡       \n1/2      '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Integral(x ** 2 * y ** 2, x, y)
    ascii_str = '  /  /           \n |  |            \n |  |  2  2      \n |  | x *y  dx dy\n |  |            \n/  /             '
    ucode_str = '⌠ ⌠            \n⎮ ⎮  2  2      \n⎮ ⎮ x ⋅y  dx dy\n⌡ ⌡            '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Integral(sin(th) / cos(ph), (th, 0, pi), (ph, 0, 2 * pi))
    ascii_str = ' 2*pi pi                           \n   /   /                           \n  |   |                            \n  |   |  sin(theta)                \n  |   |  ---------- d(theta) d(phi)\n  |   |   cos(phi)                 \n  |   |                            \n /   /                             \n 0   0                             '
    ucode_str = '2⋅π π             \n ⌠  ⌠             \n ⎮  ⎮ sin(θ)      \n ⎮  ⎮ ────── dθ dφ\n ⎮  ⎮ cos(φ)      \n ⌡  ⌡             \n 0  0             '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
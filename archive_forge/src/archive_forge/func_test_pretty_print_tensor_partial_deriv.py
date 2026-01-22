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
def test_pretty_print_tensor_partial_deriv():
    from sympy.tensor.toperators import PartialDerivative
    L = TensorIndexType('L')
    i, j, k = tensor_indices('i j k', L)
    A, B, C, D = tensor_heads('A B C D', [L])
    H = TensorHead('H', [L, L])
    expr = PartialDerivative(A(i), A(j))
    ascii_str = ' d / i\\\n---|A |\n  j\\  /\ndA     \n       '
    ucode_str = ' ∂ ⎛ i⎞\n───⎜A ⎟\n  j⎝  ⎠\n∂A     \n       '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = A(i) * PartialDerivative(H(k, -i), A(j))
    ascii_str = ' L_0  d / k   \\\nA   *---|H    |\n       j\\  L_0/\n     dA        \n               '
    ucode_str = ' L₀  ∂ ⎛ k  ⎞\nA  ⋅───⎜H   ⎟\n      j⎝  L₀⎠\n    ∂A       \n             '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = A(i) * PartialDerivative(B(k) * C(-i) + 3 * H(k, -i), A(j))
    ascii_str = ' L_0  d /   k       k     \\\nA   *---|3*H     + B *C   |\n       j\\    L_0       L_0/\n     dA                    \n                           '
    ucode_str = ' L₀  ∂ ⎛   k      k    ⎞\nA  ⋅───⎜3⋅H    + B ⋅C  ⎟\n      j⎝    L₀       L₀⎠\n    ∂A                  \n                        '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = (A(i) + B(i)) * PartialDerivative(C(j), D(j))
    ascii_str = '/ i    i\\   d  / L_0\\\n|A  + B |*-----|C   |\n\\       /   L_0\\    /\n          dD         \n                     '
    ucode_str = '⎛ i    i⎞  ∂  ⎛ L₀⎞\n⎜A  + B ⎟⋅────⎜C  ⎟\n⎝       ⎠   L₀⎝   ⎠\n          ∂D       \n                   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = (A(i) + B(i)) * PartialDerivative(C(-i), D(j))
    ascii_str = '/ L_0    L_0\\  d /    \\\n|A    + B   |*---|C   |\n\\           /   j\\ L_0/\n              dD       \n                       '
    ucode_str = '⎛ L₀    L₀⎞  ∂ ⎛   ⎞\n⎜A   + B  ⎟⋅───⎜C  ⎟\n⎝         ⎠   j⎝ L₀⎠\n            ∂D      \n                    '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = PartialDerivative(B(-i) + A(-i), A(-j), A(-n))
    ucode_str = '    2           \n   ∂   ⎛       ⎞\n───────⎜A  + B ⎟\n       ⎝ i    i⎠\n∂A  ∂A          \n  n   j         '
    assert upretty(expr) == ucode_str
    expr = PartialDerivative(3 * A(-i), A(-j), A(-n))
    ucode_str = '    2        \n   ∂   ⎛    ⎞\n───────⎜3⋅A ⎟\n       ⎝   i⎠\n∂A  ∂A       \n  n   j      '
    assert upretty(expr) == ucode_str
    expr = TensorElement(H(i, j), {i: 1})
    ascii_str = ' i=1,j\nH     \n      '
    ucode_str = ascii_str
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = TensorElement(H(i, j), {i: 1, j: 1})
    ascii_str = ' i=1,j=1\nH       \n        '
    ucode_str = ascii_str
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = TensorElement(H(i, j), {j: 1})
    ascii_str = ' i,j=1\nH     \n      '
    ucode_str = ascii_str
    expr = TensorElement(H(-i, j), {-i: 1})
    ascii_str = '    j\nH    \n i=1 '
    ucode_str = ascii_str
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
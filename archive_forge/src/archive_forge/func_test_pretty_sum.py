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
def test_pretty_sum():
    from sympy.abc import x, a, b, k, m, n
    expr = Sum(k ** k, (k, 0, n))
    ascii_str = '  n     \n ___    \n \\  `   \n  \\    k\n  /   k \n /__,   \nk = 0   '
    ucode_str = '  n     \n ___    \n ╲      \n  ╲    k\n  ╱   k \n ╱      \n ‾‾‾    \nk = 0   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(k ** k, (k, oo, n))
    ascii_str = '  n      \n ___     \n \\  `    \n  \\     k\n  /    k \n /__,    \nk = oo   '
    ucode_str = '  n     \n ___    \n ╲      \n  ╲    k\n  ╱   k \n ╱      \n ‾‾‾    \nk = ∞   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(k ** Integral(x ** n, (x, -oo, oo)), (k, 0, n ** n))
    ascii_str = '    n             \n   n              \n______            \n\\     `           \n \\        oo      \n  \\        /      \n   \\      |       \n    \\     |   n   \n     )    |  x  dx\n    /     |       \n   /     /        \n  /      -oo      \n /      k         \n/_____,           \n k = 0            '
    ucode_str = '   n            \n  n             \n______          \n╲               \n ╲              \n  ╲     ∞       \n   ╲    ⌠       \n    ╲   ⎮   n   \n    ╱   ⎮  x  dx\n   ╱    ⌡       \n  ╱     -∞      \n ╱     k        \n╱               \n‾‾‾‾‾‾          \nk = 0           '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(k ** Integral(x ** n, (x, -oo, oo)), (k, 0, Integral(x ** x, (x, -oo, oo))))
    ascii_str = ' oo                 \n  /                 \n |                  \n |   x              \n |  x  dx           \n |                  \n/                   \n-oo                 \n ______             \n \\     `            \n  \\         oo      \n   \\         /      \n    \\       |       \n     \\      |   n   \n      )     |  x  dx\n     /      |       \n    /      /        \n   /       -oo      \n  /       k         \n /_____,            \n  k = 0             '
    ucode_str = '∞                 \n⌠                 \n⎮   x             \n⎮  x  dx          \n⌡                 \n-∞                \n ______           \n ╲                \n  ╲               \n   ╲      ∞       \n    ╲     ⌠       \n     ╲    ⎮   n   \n     ╱    ⎮  x  dx\n    ╱     ⌡       \n   ╱      -∞      \n  ╱      k        \n ╱                \n ‾‾‾‾‾‾           \n k = 0            '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(k ** Integral(x ** n, (x, -oo, oo)), (k, x + n + x ** 2 + n ** 2 + x / n + 1 / x, Integral(x ** x, (x, -oo, oo))))
    ascii_str = '          oo                          \n           /                          \n          |                           \n          |   x                       \n          |  x  dx                    \n          |                           \n         /                            \n         -oo                          \n          ______                      \n          \\     `                     \n           \\                  oo      \n            \\                  /      \n             \\                |       \n              \\               |   n   \n               )              |  x  dx\n              /               |       \n             /               /        \n            /                -oo      \n           /                k         \n          /_____,                     \n     2        2       1   x           \nk = n  + n + x  + x + - + -           \n                      x   n           '
    ucode_str = '          ∞                          \n          ⌠                          \n          ⎮   x                      \n          ⎮  x  dx                   \n          ⌡                          \n          -∞                         \n           ______                    \n           ╲                         \n            ╲                        \n             ╲               ∞       \n              ╲              ⌠       \n               ╲             ⎮   n   \n               ╱             ⎮  x  dx\n              ╱              ⌡       \n             ╱               -∞      \n            ╱               k        \n           ╱                         \n           ‾‾‾‾‾‾                    \n     2        2       1   x          \nk = n  + n + x  + x + ─ + ─          \n                      x   n          '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(k ** Integral(x ** n, (x, -oo, oo)), (k, 0, x + n + x ** 2 + n ** 2 + x / n + 1 / x))
    ascii_str = ' 2        2       1   x           \nn  + n + x  + x + - + -           \n                  x   n           \n        ______                    \n        \\     `                   \n         \\                oo      \n          \\                /      \n           \\              |       \n            \\             |   n   \n             )            |  x  dx\n            /             |       \n           /             /        \n          /              -oo      \n         /              k         \n        /_____,                   \n         k = 0                    '
    ucode_str = ' 2        2       1   x          \nn  + n + x  + x + ─ + ─          \n                  x   n          \n         ______                  \n         ╲                       \n          ╲                      \n           ╲             ∞       \n            ╲            ⌠       \n             ╲           ⎮   n   \n             ╱           ⎮  x  dx\n            ╱            ⌡       \n           ╱             -∞      \n          ╱             k        \n         ╱                       \n         ‾‾‾‾‾‾                  \n         k = 0                   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(x, (x, 0, oo))
    ascii_str = '  oo   \n __    \n \\ `   \n  )   x\n /_,   \nx = 0  '
    ucode_str = '  ∞    \n ___   \n ╲     \n  ╲    \n  ╱   x\n ╱     \n ‾‾‾   \nx = 0  '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(x ** 2, (x, 0, oo))
    ascii_str = '  oo    \n ___    \n \\  `   \n  \\    2\n  /   x \n /__,   \nx = 0   '
    ucode_str = '  ∞     \n ___    \n ╲      \n  ╲    2\n  ╱   x \n ╱      \n ‾‾‾    \nx = 0   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(x / 2, (x, 0, oo))
    ascii_str = '  oo   \n ___   \n \\  `  \n  \\   x\n   )  -\n  /   2\n /__,  \nx = 0  '
    ucode_str = '  ∞    \n ____  \n ╲     \n  ╲    \n   ╲  x\n   ╱  ─\n  ╱   2\n ╱     \n ‾‾‾‾  \nx = 0  '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(x ** 3 / 2, (x, 0, oo))
    ascii_str = '  oo    \n____    \n\\   `   \n \\     3\n  \\   x \n  /   --\n /    2 \n/___,   \nx = 0   '
    ucode_str = '  ∞     \n ____   \n ╲      \n  ╲    3\n   ╲  x \n   ╱  ──\n  ╱   2 \n ╱      \n ‾‾‾‾   \nx = 0   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum((x ** 3 * y ** (x / 2)) ** n, (x, 0, oo))
    ascii_str = '  oo          \n____          \n\\   `         \n \\           n\n  \\   /    x\\ \n   )  |    -| \n  /   | 3  2| \n /    \\x *y / \n/___,         \nx = 0         '
    ucode_str = '  ∞           \n_____         \n╲             \n ╲            \n  ╲          n\n   ╲  ⎛    x⎞ \n   ╱  ⎜    ─⎟ \n  ╱   ⎜ 3  2⎟ \n ╱    ⎝x ⋅y ⎠ \n╱             \n‾‾‾‾‾         \nx = 0         '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(1 / x ** 2, (x, 0, oo))
    ascii_str = '  oo    \n____    \n\\   `   \n \\    1 \n  \\   --\n  /    2\n /    x \n/___,   \nx = 0   '
    ucode_str = '  ∞     \n ____   \n ╲      \n  ╲   1 \n   ╲  ──\n   ╱   2\n  ╱   x \n ╱      \n ‾‾‾‾   \nx = 0   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(1 / y ** (a / b), (x, 0, oo))
    ascii_str = '  oo      \n____      \n\\   `     \n \\     -a \n  \\    ---\n  /     b \n /    y   \n/___,     \nx = 0     '
    ucode_str = '  ∞       \n ____     \n ╲        \n  ╲    -a \n   ╲   ───\n   ╱    b \n  ╱   y   \n ╱        \n ‾‾‾‾     \nx = 0     '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Sum(1 / y ** (a / b), (x, 0, oo), (y, 1, 2))
    ascii_str = '  2     oo     \n____  ____     \n\\   ` \\   `    \n \\     \\     -a\n  \\     \\    --\n  /     /    b \n /     /    y  \n/___, /___,    \ny = 1 x = 0    '
    ucode_str = '  2     ∞      \n____  ____     \n╲     ╲        \n ╲     ╲     -a\n  ╲     ╲    ──\n  ╱     ╱    b \n ╱     ╱    y  \n╱     ╱        \n‾‾‾‾  ‾‾‾‾     \ny = 1 x = 0    '
    expr = Sum(1 / (1 + 1 / (1 + 1 / k)) + 1, (k, 111, 1 + 1 / n), (k, 1 / (1 + m), oo)) + 1 / (1 + 1 / k)
    ascii_str = '               1                         \n           1 + -                         \n    oo         n                         \n  _____    _____                         \n  \\    `   \\    `                        \n   \\        \\     /        1    \\        \n    \\        \\    |1 + ---------|        \n     \\        \\   |          1  |     1  \n      )        )  |    1 + -----| + -----\n     /        /   |            1|       1\n    /        /    |        1 + -|   1 + -\n   /        /     \\            k/       k\n  /____,   /____,                        \n      1   k = 111                        \nk = -----                                \n    m + 1                                '
    ucode_str = '               1                         \n           1 + ─                         \n    ∞          n                         \n  ______   ______                        \n  ╲        ╲                             \n   ╲        ╲                            \n    ╲        ╲    ⎛        1    ⎞        \n     ╲        ╲   ⎜1 + ─────────⎟        \n      ╲        ╲  ⎜          1  ⎟     1  \n      ╱        ╱  ⎜    1 + ─────⎟ + ─────\n     ╱        ╱   ⎜            1⎟       1\n    ╱        ╱    ⎜        1 + ─⎟   1 + ─\n   ╱        ╱     ⎝            k⎠       k\n  ╱        ╱                             \n  ‾‾‾‾‾‾   ‾‾‾‾‾‾                        \n      1   k = 111                        \nk = ─────                                \n    m + 1                                '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
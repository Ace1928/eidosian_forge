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
def test_pretty_ndim_arrays():
    x, y, z, w = symbols('x y z w')
    for ArrayType in (ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableDenseNDimArray, MutableSparseNDimArray):
        M = ArrayType(x)
        assert pretty(M) == 'x'
        assert upretty(M) == 'x'
        M = ArrayType([[1 / x, y], [z, w]])
        M1 = ArrayType([1 / x, y, z])
        M2 = tensorproduct(M1, M)
        M3 = tensorproduct(M, M)
        ascii_str = '[1   ]\n[-  y]\n[x   ]\n[    ]\n[z  w]'
        ucode_str = '⎡1   ⎤\n⎢─  y⎥\n⎢x   ⎥\n⎢    ⎥\n⎣z  w⎦'
        assert pretty(M) == ascii_str
        assert upretty(M) == ucode_str
        ascii_str = '[1      ]\n[-  y  z]\n[x      ]'
        ucode_str = '⎡1      ⎤\n⎢─  y  z⎥\n⎣x      ⎦'
        assert pretty(M1) == ascii_str
        assert upretty(M1) == ucode_str
        ascii_str = '[[1   y]                       ]\n[[--  -]              [z      ]]\n[[ 2  x]  [ y    2 ]  [-   y*z]]\n[[x    ]  [ -   y  ]  [x      ]]\n[[     ]  [ x      ]  [       ]]\n[[z   w]  [        ]  [ 2     ]]\n[[-   -]  [y*z  w*y]  [z   w*z]]\n[[x   x]                       ]'
        ucode_str = '⎡⎡1   y⎤                       ⎤\n⎢⎢──  ─⎥              ⎡z      ⎤⎥\n⎢⎢ 2  x⎥  ⎡ y    2 ⎤  ⎢─   y⋅z⎥⎥\n⎢⎢x    ⎥  ⎢ ─   y  ⎥  ⎢x      ⎥⎥\n⎢⎢     ⎥  ⎢ x      ⎥  ⎢       ⎥⎥\n⎢⎢z   w⎥  ⎢        ⎥  ⎢ 2     ⎥⎥\n⎢⎢─   ─⎥  ⎣y⋅z  w⋅y⎦  ⎣z   w⋅z⎦⎥\n⎣⎣x   x⎦                       ⎦'
        assert pretty(M2) == ascii_str
        assert upretty(M2) == ucode_str
        ascii_str = '[ [1   y]             ]\n[ [--  -]             ]\n[ [ 2  x]   [ y    2 ]]\n[ [x    ]   [ -   y  ]]\n[ [     ]   [ x      ]]\n[ [z   w]   [        ]]\n[ [-   -]   [y*z  w*y]]\n[ [x   x]             ]\n[                     ]\n[[z      ]  [ w      ]]\n[[-   y*z]  [ -   w*y]]\n[[x      ]  [ x      ]]\n[[       ]  [        ]]\n[[ 2     ]  [      2 ]]\n[[z   w*z]  [w*z  w  ]]'
        ucode_str = '⎡ ⎡1   y⎤             ⎤\n⎢ ⎢──  ─⎥             ⎥\n⎢ ⎢ 2  x⎥   ⎡ y    2 ⎤⎥\n⎢ ⎢x    ⎥   ⎢ ─   y  ⎥⎥\n⎢ ⎢     ⎥   ⎢ x      ⎥⎥\n⎢ ⎢z   w⎥   ⎢        ⎥⎥\n⎢ ⎢─   ─⎥   ⎣y⋅z  w⋅y⎦⎥\n⎢ ⎣x   x⎦             ⎥\n⎢                     ⎥\n⎢⎡z      ⎤  ⎡ w      ⎤⎥\n⎢⎢─   y⋅z⎥  ⎢ ─   w⋅y⎥⎥\n⎢⎢x      ⎥  ⎢ x      ⎥⎥\n⎢⎢       ⎥  ⎢        ⎥⎥\n⎢⎢ 2     ⎥  ⎢      2 ⎥⎥\n⎣⎣z   w⋅z⎦  ⎣w⋅z  w  ⎦⎦'
        assert pretty(M3) == ascii_str
        assert upretty(M3) == ucode_str
        Mrow = ArrayType([[x, y, 1 / z]])
        Mcolumn = ArrayType([[x], [y], [1 / z]])
        Mcol2 = ArrayType([Mcolumn.tolist()])
        ascii_str = '[[      1]]\n[[x  y  -]]\n[[      z]]'
        ucode_str = '⎡⎡      1⎤⎤\n⎢⎢x  y  ─⎥⎥\n⎣⎣      z⎦⎦'
        assert pretty(Mrow) == ascii_str
        assert upretty(Mrow) == ucode_str
        ascii_str = '[x]\n[ ]\n[y]\n[ ]\n[1]\n[-]\n[z]'
        ucode_str = '⎡x⎤\n⎢ ⎥\n⎢y⎥\n⎢ ⎥\n⎢1⎥\n⎢─⎥\n⎣z⎦'
        assert pretty(Mcolumn) == ascii_str
        assert upretty(Mcolumn) == ucode_str
        ascii_str = '[[x]]\n[[ ]]\n[[y]]\n[[ ]]\n[[1]]\n[[-]]\n[[z]]'
        ucode_str = '⎡⎡x⎤⎤\n⎢⎢ ⎥⎥\n⎢⎢y⎥⎥\n⎢⎢ ⎥⎥\n⎢⎢1⎥⎥\n⎢⎢─⎥⎥\n⎣⎣z⎦⎦'
        assert pretty(Mcol2) == ascii_str
        assert upretty(Mcol2) == ucode_str
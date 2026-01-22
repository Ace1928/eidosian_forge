from collections import defaultdict
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core import (Basic, S, Add, Mul, Pow, Symbol, sympify,
from sympy.core.exprtools import factor_nc
from sympy.core.parameters import global_parameters
from sympy.core.function import (expand_log, count_ops, _mexpand,
from sympy.core.numbers import Float, I, pi, Rational
from sympy.core.relational import Relational
from sympy.core.rules import Transform
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympify
from sympy.core.traversal import bottom_up as _bottom_up, walk as _walk
from sympy.functions import gamma, exp, sqrt, log, exp_polar, re
from sympy.functions.combinatorial.factorials import CombinatorialFunction
from sympy.functions.elementary.complexes import unpolarify, Abs, sign
from sympy.functions.elementary.exponential import ExpBase
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import (Piecewise, piecewise_fold,
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.functions.special.bessel import (BesselBase, besselj, besseli,
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import Integral
from sympy.matrices.expressions import (MatrixExpr, MatAdd, MatMul,
from sympy.polys import together, cancel, factor
from sympy.polys.numberfields.minpoly import _is_sum_surds, _minimal_polynomial_sq
from sympy.simplify.combsimp import combsimp
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import radsimp, fraction, collect_abs
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.trigsimp import trigsimp, exptrigsimp
from sympy.utilities.decorator import deprecated
from sympy.utilities.iterables import has_variety, sift, subsets, iterable
from sympy.utilities.misc import as_int
import mpmath
def product_mul(self, other, method=0):
    """Helper function for Product simplification"""
    if type(self) is type(other):
        if method == 0:
            if self.limits == other.limits:
                return Product(self.function * other.function, *self.limits)
        elif method == 1:
            if simplify(self.function - other.function) == 0:
                if len(self.limits) == len(other.limits) == 1:
                    i = self.limits[0][0]
                    x1 = self.limits[0][1]
                    y1 = self.limits[0][2]
                    j = other.limits[0][0]
                    x2 = other.limits[0][1]
                    y2 = other.limits[0][2]
                    if i == j:
                        if x2 == y1 + 1:
                            return Product(self.function, (i, x1, y2))
                        elif x1 == y2 + 1:
                            return Product(self.function, (i, x2, y1))
    return Mul(self, other)
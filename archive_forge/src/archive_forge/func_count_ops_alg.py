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
def count_ops_alg(expr):
    """Optimized count algebraic operations with no recursion into
        non-algebraic args that ``core.function.count_ops`` does. Also returns
        whether rational functions may be present according to negative
        exponents of powers or non-number fractions.

        Returns
        =======

        ops, ratfunc : int, bool
            ``ops`` is the number of algebraic operations starting at the top
            level expression (not recursing into non-alg children). ``ratfunc``
            specifies whether the expression MAY contain rational functions
            which ``cancel`` MIGHT optimize.
        """
    ops = 0
    args = [expr]
    ratfunc = False
    while args:
        a = args.pop()
        if not isinstance(a, Basic):
            continue
        if a.is_Rational:
            if a is not S.One:
                ops += bool(a.p < 0) + bool(a.q != 1)
        elif a.is_Mul:
            if a.could_extract_minus_sign():
                ops += 1
                if a.args[0] is S.NegativeOne:
                    a = a.as_two_terms()[1]
                else:
                    a = -a
            n, d = fraction(a)
            if n.is_Integer:
                ops += 1 + bool(n < 0)
                args.append(d)
            elif d is not S.One:
                if not d.is_Integer:
                    args.append(d)
                    ratfunc = True
                ops += 1
                args.append(n)
            else:
                ops += len(a.args) - 1
                args.extend(a.args)
        elif a.is_Add:
            laargs = len(a.args)
            negs = 0
            for ai in a.args:
                if ai.could_extract_minus_sign():
                    negs += 1
                    ai = -ai
                args.append(ai)
            ops += laargs - (negs != laargs)
        elif a.is_Pow:
            ops += 1
            args.append(a.base)
            if not ratfunc:
                ratfunc = a.exp.is_negative is not False
    return (ops, ratfunc)
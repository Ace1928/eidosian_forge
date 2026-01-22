from typing import Tuple as tTuple
from sympy.calculus.singularities import is_decreasing
from sympy.calculus.accumulationbounds import AccumulationBounds
from .expr_with_intlimits import ExprWithIntLimits
from .expr_with_limits import AddWithLimits
from .gosper import gosper_sum
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Derivative, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Float, _illegal
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Wild, Symbol, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cot, csc
from sympy.functions.special.hyper import hyper
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import PolynomialError, PolificationFailed
from sympy.polys.polytools import parallel_poly_from_expr, Poly, factor
from sympy.polys.rationaltools import together
from sympy.series.limitseq import limit_seq
from sympy.series.order import O
from sympy.series.residues import residue
from sympy.sets.sets import FiniteSet, Interval
from sympy.utilities.iterables import sift
import itertools
def eval_sum(f, limits):
    i, a, b = limits
    if f.is_zero:
        return S.Zero
    if i not in f.free_symbols:
        return f * (b - a + 1)
    if a == b:
        return f.subs(i, a)
    if isinstance(f, Piecewise):
        if not any((i in arg.args[1].free_symbols for arg in f.args)):
            newargs = []
            for arg in f.args:
                newexpr = eval_sum(arg.expr, limits)
                if newexpr is None:
                    return None
                newargs.append((newexpr, arg.cond))
            return f.func(*newargs)
    if f.has(KroneckerDelta):
        from .delta import deltasummation, _has_simple_delta
        f = f.replace(lambda x: isinstance(x, Sum), lambda x: x.factor())
        if _has_simple_delta(f, limits[0]):
            return deltasummation(f, limits)
    dif = b - a
    definite = dif.is_Integer
    if definite and dif < 100:
        return eval_sum_direct(f, (i, a, b))
    if isinstance(f, Piecewise):
        return None
    value = eval_sum_symbolic(f.expand(), (i, a, b))
    if value is not None:
        return value
    if definite:
        return eval_sum_direct(f, (i, a, b))
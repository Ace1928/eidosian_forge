from collections import defaultdict
from functools import reduce
from math import prod
from sympy.core.function import expand_log, count_ops, _coeff_isneg
from sympy.core import sympify, Basic, Dummy, S, Add, Mul, Pow, expand_mul, factor_terms
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.numbers import Integer, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.rules import Transform
from sympy.functions import exp_polar, exp, log, root, polarify, unpolarify
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys import lcm, gcd
from sympy.ntheory.factor_ import multiplicity
def bkey(b, e=None):
    """Return (b**s, c.q), c.p where e -> c*s. If e is not given then
            it will be taken by using as_base_exp() on the input b.
            e.g.
                x**3/2 -> (x, 2), 3
                x**y -> (x**y, 1), 1
                x**(2*y/3) -> (x**y, 3), 2
                exp(x/2) -> (exp(a), 2), 1

            """
    if e is not None:
        if e.is_Integer:
            return ((b, S.One), e)
        elif e.is_Rational:
            return ((b, Integer(e.q)), Integer(e.p))
        else:
            c, m = e.as_coeff_Mul(rational=True)
            if c is not S.One:
                if m.is_integer:
                    return ((b, Integer(c.q)), m * Integer(c.p))
                return ((b ** m, Integer(c.q)), Integer(c.p))
            else:
                return ((b ** e, S.One), S.One)
    else:
        return bkey(*b.as_base_exp())
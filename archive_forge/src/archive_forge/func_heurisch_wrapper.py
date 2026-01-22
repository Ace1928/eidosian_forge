from __future__ import annotations
from itertools import permutations
from functools import reduce
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import Wild, Dummy, Symbol
from sympy.core.basic import sympify
from sympy.core.numbers import Rational, pi, I
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.traversal import iterfreeargs
from sympy.functions import exp, sin, cos, tan, cot, asin, atan
from sympy.functions import log, sinh, cosh, tanh, coth, asinh
from sympy.functions import sqrt, erf, erfi, li, Ei
from sympy.functions import besselj, bessely, besseli, besselk
from sympy.functions import hankel1, hankel2, jn, yn
from sympy.functions.elementary.complexes import Abs, re, im, sign, arg
from sympy.functions.elementary.exponential import LambertW
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.simplify.radsimp import collect
from sympy.logic.boolalg import And, Or
from sympy.utilities.iterables import uniq
from sympy.polys import quo, gcd, lcm, factor_list, cancel, PolynomialError
from sympy.polys.monomials import itermonomials
from sympy.polys.polyroots import root_factors
from sympy.polys.rings import PolyRing
from sympy.polys.solvers import solve_lin_sys
from sympy.polys.constructor import construct_domain
from sympy.integrals.integrals import integrate
def heurisch_wrapper(f, x, rewrite=False, hints=None, mappings=None, retries=3, degree_offset=0, unnecessary_permutations=None, _try_heurisch=None):
    """
    A wrapper around the heurisch integration algorithm.

    Explanation
    ===========

    This method takes the result from heurisch and checks for poles in the
    denominator. For each of these poles, the integral is reevaluated, and
    the final integration result is given in terms of a Piecewise.

    Examples
    ========

    >>> from sympy import cos, symbols
    >>> from sympy.integrals.heurisch import heurisch, heurisch_wrapper
    >>> n, x = symbols('n x')
    >>> heurisch(cos(n*x), x)
    sin(n*x)/n
    >>> heurisch_wrapper(cos(n*x), x)
    Piecewise((sin(n*x)/n, Ne(n, 0)), (x, True))

    See Also
    ========

    heurisch
    """
    from sympy.solvers.solvers import solve, denoms
    f = sympify(f)
    if not f.has_free(x):
        return f * x
    res = heurisch(f, x, rewrite, hints, mappings, retries, degree_offset, unnecessary_permutations, _try_heurisch)
    if not isinstance(res, Basic):
        return res
    slns = []
    for d in ordered(denoms(res)):
        try:
            slns += solve([d], dict=True, exclude=(x,))
        except NotImplementedError:
            pass
    if not slns:
        return res
    slns = list(uniq(slns))
    slns0 = []
    for d in denoms(f):
        try:
            slns0 += solve([d], dict=True, exclude=(x,))
        except NotImplementedError:
            pass
    slns = [s for s in slns if s not in slns0]
    if not slns:
        return res
    if len(slns) > 1:
        eqs = []
        for sub_dict in slns:
            eqs.extend([Eq(key, value) for key, value in sub_dict.items()])
        slns = solve(eqs, dict=True, exclude=(x,)) + slns
    pairs = []
    for sub_dict in slns:
        expr = heurisch(f.subs(sub_dict), x, rewrite, hints, mappings, retries, degree_offset, unnecessary_permutations, _try_heurisch)
        cond = And(*[Eq(key, value) for key, value in sub_dict.items()])
        generic = Or(*[Ne(key, value) for key, value in sub_dict.items()])
        if expr is None:
            expr = integrate(f.subs(sub_dict), x)
        pairs.append((expr, cond))
    if len(pairs) == 1:
        pairs = [(heurisch(f, x, rewrite, hints, mappings, retries, degree_offset, unnecessary_permutations, _try_heurisch), generic), (pairs[0][0], True)]
    else:
        pairs.append((heurisch(f, x, rewrite, hints, mappings, retries, degree_offset, unnecessary_permutations, _try_heurisch), True))
    return Piecewise(*pairs)
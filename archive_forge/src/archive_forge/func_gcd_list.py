from functools import wraps, reduce
from operator import mul
from typing import Optional
from sympy.core import (
from sympy.core.basic import Basic
from sympy.core.decorators import _sympifyit
from sympy.core.exprtools import Factors, factor_nc, factor_terms
from sympy.core.evalf import (
from sympy.core.function import Derivative
from sympy.core.mul import Mul, _keep_coeff
from sympy.core.numbers import ilcm, I, Integer, equal_valued
from sympy.core.relational import Relational, Equality
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal, bottom_up
from sympy.logic.boolalg import BooleanAtom
from sympy.polys import polyoptions as options
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (
from sympy.polys.polyutils import (
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.utilities import group, public, filldedent
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, sift
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence
@public
def gcd_list(seq, *gens, **args):
    """
    Compute GCD of a list of polynomials.

    Examples
    ========

    >>> from sympy import gcd_list
    >>> from sympy.abc import x

    >>> gcd_list([x**3 - 1, x**2 - 1, x**2 - 3*x + 2])
    x - 1

    """
    seq = sympify(seq)

    def try_non_polynomial_gcd(seq):
        if not gens and (not args):
            domain, numbers = construct_domain(seq)
            if not numbers:
                return domain.zero
            elif domain.is_Numerical:
                result, numbers = (numbers[0], numbers[1:])
                for number in numbers:
                    result = domain.gcd(result, number)
                    if domain.is_one(result):
                        break
                return domain.to_sympy(result)
        return None
    result = try_non_polynomial_gcd(seq)
    if result is not None:
        return result
    options.allowed_flags(args, ['polys'])
    try:
        polys, opt = parallel_poly_from_expr(seq, *gens, **args)
        if len(seq) > 1 and all((elt.is_algebraic and elt.is_irrational for elt in seq)):
            a = seq[-1]
            lst = [(a / elt).ratsimp() for elt in seq[:-1]]
            if all((frc.is_rational for frc in lst)):
                lc = 1
                for frc in lst:
                    lc = lcm(lc, frc.as_numer_denom()[0])
                return abs(a / lc)
    except PolificationFailed as exc:
        result = try_non_polynomial_gcd(exc.exprs)
        if result is not None:
            return result
        else:
            raise ComputationFailed('gcd_list', len(seq), exc)
    if not polys:
        if not opt.polys:
            return S.Zero
        else:
            return Poly(0, opt=opt)
    result, polys = (polys[0], polys[1:])
    for poly in polys:
        result = result.gcd(poly)
        if result.is_one:
            break
    if not opt.polys:
        return result.as_expr()
    else:
        return result
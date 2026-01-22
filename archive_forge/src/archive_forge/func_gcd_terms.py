from .add import Add
from .mul import Mul, _keep_coeff
from .power import Pow
from .basic import Basic
from .expr import Expr
from .function import expand_power_exp
from .sympify import sympify
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy
from .traversal import preorder_traversal
from .coreerrors import NonCommutativeExpression
from .containers import Tuple, Dict
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import (common_prefix, common_suffix,
from collections import defaultdict
from typing import Tuple as tTuple
def gcd_terms(terms, isprimitive=False, clear=True, fraction=True):
    """Compute the GCD of ``terms`` and put them together.

    Parameters
    ==========

    terms : Expr
        Can be an expression or a non-Basic sequence of expressions
        which will be handled as though they are terms from a sum.

    isprimitive : bool, optional
        If ``isprimitive`` is True the _gcd_terms will not run the primitive
        method on the terms.

    clear : bool, optional
        It controls the removal of integers from the denominator of an Add
        expression. When True (default), all numerical denominator will be cleared;
        when False the denominators will be cleared only if all terms had numerical
        denominators other than 1.

    fraction : bool, optional
        When True (default), will put the expression over a common
        denominator.

    Examples
    ========

    >>> from sympy import gcd_terms
    >>> from sympy.abc import x, y

    >>> gcd_terms((x + 1)**2*y + (x + 1)*y**2)
    y*(x + 1)*(x + y + 1)
    >>> gcd_terms(x/2 + 1)
    (x + 2)/2
    >>> gcd_terms(x/2 + 1, clear=False)
    x/2 + 1
    >>> gcd_terms(x/2 + y/2, clear=False)
    (x + y)/2
    >>> gcd_terms(x/2 + 1/x)
    (x**2 + 2)/(2*x)
    >>> gcd_terms(x/2 + 1/x, fraction=False)
    (x + 2/x)/2
    >>> gcd_terms(x/2 + 1/x, fraction=False, clear=False)
    x/2 + 1/x

    >>> gcd_terms(x/2/y + 1/x/y)
    (x**2 + 2)/(2*x*y)
    >>> gcd_terms(x/2/y + 1/x/y, clear=False)
    (x**2/2 + 1)/(x*y)
    >>> gcd_terms(x/2/y + 1/x/y, clear=False, fraction=False)
    (x/2 + 1/x)/y

    The ``clear`` flag was ignored in this case because the returned
    expression was a rational expression, not a simple sum.

    See Also
    ========

    factor_terms, sympy.polys.polytools.terms_gcd

    """

    def mask(terms):
        """replace nc portions of each term with a unique Dummy symbols
        and return the replacements to restore them"""
        args = [(a, []) if a.is_commutative else a.args_cnc() for a in terms]
        reps = []
        for i, (c, nc) in enumerate(args):
            if nc:
                nc = Mul(*nc)
                d = Dummy()
                reps.append((d, nc))
                c.append(d)
                args[i] = Mul(*c)
            else:
                args[i] = c
        return (args, dict(reps))
    isadd = isinstance(terms, Add)
    addlike = isadd or (not isinstance(terms, Basic) and is_sequence(terms, include=set) and (not isinstance(terms, Dict)))
    if addlike:
        if isadd:
            terms = list(terms.args)
        else:
            terms = sympify(terms)
        terms, reps = mask(terms)
        cont, numer, denom = _gcd_terms(terms, isprimitive, fraction)
        numer = numer.xreplace(reps)
        coeff, factors = cont.as_coeff_Mul()
        if not clear:
            c, _coeff = coeff.as_coeff_Mul()
            if not c.is_Integer and (not clear) and numer.is_Add:
                n, d = c.as_numer_denom()
                _numer = numer / d
                if any((a.as_coeff_Mul()[0].is_Integer for a in _numer.args)):
                    numer = _numer
                    coeff = n * _coeff
        return _keep_coeff(coeff, factors * numer / denom, clear=clear)
    if not isinstance(terms, Basic):
        return terms
    if terms.is_Atom:
        return terms
    if terms.is_Mul:
        c, args = terms.as_coeff_mul()
        return _keep_coeff(c, Mul(*[gcd_terms(i, isprimitive, clear, fraction) for i in args]), clear=clear)

    def handle(a):
        if not isinstance(a, Expr):
            if isinstance(a, Basic):
                if not a.args:
                    return a
                return a.func(*[handle(i) for i in a.args])
            return type(a)([handle(i) for i in a])
        return gcd_terms(a, isprimitive, clear, fraction)
    if isinstance(terms, Dict):
        return Dict(*[(k, handle(v)) for k, v in terms.args])
    return terms.func(*[handle(i) for i in terms.args])
from collections import defaultdict
from functools import reduce
from sympy.core import (sympify, Basic, S, Expr, factor_terms,
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
from sympy.core.numbers import I, Integer, igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
def trigsimp_groebner(expr, hints=[], quick=False, order='grlex', polynomial=False):
    """
    Simplify trigonometric expressions using a groebner basis algorithm.

    Explanation
    ===========

    This routine takes a fraction involving trigonometric or hyperbolic
    expressions, and tries to simplify it. The primary metric is the
    total degree. Some attempts are made to choose the simplest possible
    expression of the minimal degree, but this is non-rigorous, and also
    very slow (see the ``quick=True`` option).

    If ``polynomial`` is set to True, instead of simplifying numerator and
    denominator together, this function just brings numerator and denominator
    into a canonical form. This is much faster, but has potentially worse
    results. However, if the input is a polynomial, then the result is
    guaranteed to be an equivalent polynomial of minimal degree.

    The most important option is hints. Its entries can be any of the
    following:

    - a natural number
    - a function
    - an iterable of the form (func, var1, var2, ...)
    - anything else, interpreted as a generator

    A number is used to indicate that the search space should be increased.
    A function is used to indicate that said function is likely to occur in a
    simplified expression.
    An iterable is used indicate that func(var1 + var2 + ...) is likely to
    occur in a simplified .
    An additional generator also indicates that it is likely to occur.
    (See examples below).

    This routine carries out various computationally intensive algorithms.
    The option ``quick=True`` can be used to suppress one particularly slow
    step (at the expense of potentially more complicated results, but never at
    the expense of increased total degree).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import sin, tan, cos, sinh, cosh, tanh
    >>> from sympy.simplify.trigsimp import trigsimp_groebner

    Suppose you want to simplify ``sin(x)*cos(x)``. Naively, nothing happens:

    >>> ex = sin(x)*cos(x)
    >>> trigsimp_groebner(ex)
    sin(x)*cos(x)

    This is because ``trigsimp_groebner`` only looks for a simplification
    involving just ``sin(x)`` and ``cos(x)``. You can tell it to also try
    ``2*x`` by passing ``hints=[2]``:

    >>> trigsimp_groebner(ex, hints=[2])
    sin(2*x)/2
    >>> trigsimp_groebner(sin(x)**2 - cos(x)**2, hints=[2])
    -cos(2*x)

    Increasing the search space this way can quickly become expensive. A much
    faster way is to give a specific expression that is likely to occur:

    >>> trigsimp_groebner(ex, hints=[sin(2*x)])
    sin(2*x)/2

    Hyperbolic expressions are similarly supported:

    >>> trigsimp_groebner(sinh(2*x)/sinh(x))
    2*cosh(x)

    Note how no hints had to be passed, since the expression already involved
    ``2*x``.

    The tangent function is also supported. You can either pass ``tan`` in the
    hints, to indicate that tan should be tried whenever cosine or sine are,
    or you can pass a specific generator:

    >>> trigsimp_groebner(sin(x)/cos(x), hints=[tan])
    tan(x)
    >>> trigsimp_groebner(sinh(x)/cosh(x), hints=[tanh(x)])
    tanh(x)

    Finally, you can use the iterable form to suggest that angle sum formulae
    should be tried:

    >>> ex = (tan(x) + tan(y))/(1 - tan(x)*tan(y))
    >>> trigsimp_groebner(ex, hints=[(tan, x, y)])
    tan(x + y)
    """

    def parse_hints(hints):
        """Split hints into (n, funcs, iterables, gens)."""
        n = 1
        funcs, iterables, gens = ([], [], [])
        for e in hints:
            if isinstance(e, (SYMPY_INTS, Integer)):
                n = e
            elif isinstance(e, FunctionClass):
                funcs.append(e)
            elif iterable(e):
                iterables.append((e[0], e[1:]))
                gens.extend(parallel_poly_from_expr([e[0](x) for x in e[1:]] + [e[0](Add(*e[1:]))])[1].gens)
            else:
                gens.append(e)
        return (n, funcs, iterables, gens)

    def build_ideal(x, terms):
        """
        Build generators for our ideal. ``Terms`` is an iterable with elements of
        the form (fn, coeff), indicating that we have a generator fn(coeff*x).

        If any of the terms is trigonometric, sin(x) and cos(x) are guaranteed
        to appear in terms. Similarly for hyperbolic functions. For tan(n*x),
        sin(n*x) and cos(n*x) are guaranteed.
        """
        I = []
        y = Dummy('y')
        for fn, coeff in terms:
            for c, s, t, rel in ([cos, sin, tan, cos(x) ** 2 + sin(x) ** 2 - 1], [cosh, sinh, tanh, cosh(x) ** 2 - sinh(x) ** 2 - 1]):
                if coeff == 1 and fn in [c, s]:
                    I.append(rel)
                elif fn == t:
                    I.append(t(coeff * x) * c(coeff * x) - s(coeff * x))
                elif fn in [c, s]:
                    cn = fn(coeff * y).expand(trig=True).subs(y, x)
                    I.append(fn(coeff * x) - cn)
        return list(set(I))

    def analyse_gens(gens, hints):
        """
        Analyse the generators ``gens``, using the hints ``hints``.

        The meaning of ``hints`` is described in the main docstring.
        Return a new list of generators, and also the ideal we should
        work with.
        """
        n, funcs, iterables, extragens = parse_hints(hints)
        debug('n=%s   funcs: %s   iterables: %s    extragens: %s', (funcs, iterables, extragens))
        gens = list(gens)
        gens.extend(extragens)
        funcs = list(set(funcs))
        iterables = list(set(iterables))
        gens = list(set(gens))
        allfuncs = {sin, cos, tan, sinh, cosh, tanh}
        trigterms = [(g.args[0].as_coeff_mul(), g.func) for g in gens if g.func in allfuncs]
        freegens = [g for g in gens if g.func not in allfuncs]
        newgens = []
        trigdict = {}
        for (coeff, var), fn in trigterms:
            trigdict.setdefault(var, []).append((coeff, fn))
        res = []
        for key, val in trigdict.items():
            fns = [x[1] for x in val]
            val = [x[0] for x in val]
            gcd = reduce(igcd, val)
            terms = [(fn, v / gcd) for fn, v in zip(fns, val)]
            fs = set(funcs + fns)
            for c, s, t in ([cos, sin, tan], [cosh, sinh, tanh]):
                if any((x in fs for x in (c, s, t))):
                    fs.add(c)
                    fs.add(s)
            for fn in fs:
                for k in range(1, n + 1):
                    terms.append((fn, k))
            extra = []
            for fn, v in terms:
                if fn == tan:
                    extra.append((sin, v))
                    extra.append((cos, v))
                if fn in [sin, cos] and tan in fs:
                    extra.append((tan, v))
                if fn == tanh:
                    extra.append((sinh, v))
                    extra.append((cosh, v))
                if fn in [sinh, cosh] and tanh in fs:
                    extra.append((tanh, v))
            terms.extend(extra)
            x = gcd * Mul(*key)
            r = build_ideal(x, terms)
            res.extend(r)
            newgens.extend({fn(v * x) for fn, v in terms})
        for fn, args in iterables:
            if fn == tan:
                iterables.extend([(sin, args), (cos, args)])
            elif fn == tanh:
                iterables.extend([(sinh, args), (cosh, args)])
            else:
                dummys = symbols('d:%i' % len(args), cls=Dummy)
                expr = fn(Add(*dummys)).expand(trig=True).subs(list(zip(dummys, args)))
                res.append(fn(Add(*args)) - expr)
        if myI in gens:
            res.append(myI ** 2 + 1)
            freegens.remove(myI)
            newgens.append(myI)
        return (res, freegens, newgens)
    myI = Dummy('I')
    expr = expr.subs(S.ImaginaryUnit, myI)
    subs = [(myI, S.ImaginaryUnit)]
    num, denom = cancel(expr).as_numer_denom()
    try:
        (pnum, pdenom), opt = parallel_poly_from_expr([num, denom])
    except PolificationFailed:
        return expr
    debug('initial gens:', opt.gens)
    ideal, freegens, gens = analyse_gens(opt.gens, hints)
    debug('ideal:', ideal)
    debug('new gens:', gens, ' -- len', len(gens))
    debug('free gens:', freegens, ' -- len', len(gens))
    if not gens:
        return expr
    G = groebner(ideal, order=order, gens=gens, domain=ZZ)
    debug('groebner basis:', list(G), ' -- len', len(G))
    from sympy.simplify.ratsimp import ratsimpmodprime
    if freegens and pdenom.has_only_gens(*set(gens).intersection(pdenom.gens)):
        num = Poly(num, gens=gens + freegens).eject(*gens)
        res = []
        for monom, coeff in num.terms():
            ourgens = set(parallel_poly_from_expr([coeff, denom])[1].gens)
            changed = True
            while changed:
                changed = False
                for p in ideal:
                    p = Poly(p)
                    if not ourgens.issuperset(p.gens) and (not p.has_only_gens(*set(p.gens).difference(ourgens))):
                        changed = True
                        ourgens.update(p.exclude().gens)
            realgens = [x for x in gens if x in ourgens]
            ourG = [g.as_expr() for g in G.polys if g.has_only_gens(*ourgens.intersection(g.gens))]
            res.append(Mul(*[a ** b for a, b in zip(freegens, monom)]) * ratsimpmodprime(coeff / denom, ourG, order=order, gens=realgens, quick=quick, domain=ZZ, polynomial=polynomial).subs(subs))
        return Add(*res)
        return Add(*[Mul(*[a ** b for a, b in zip(freegens, monom)]) * ratsimpmodprime(coeff / denom, list(G), order=order, gens=gens, quick=quick, domain=ZZ) for monom, coeff in num.terms()])
    else:
        return ratsimpmodprime(expr, list(G), order=order, gens=freegens + gens, quick=quick, domain=ZZ, polynomial=polynomial).subs(subs)
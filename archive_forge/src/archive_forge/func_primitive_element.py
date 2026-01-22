from sympy.core.add import Add
from sympy.core.numbers import AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.ntheory import sieve
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import QQ
from sympy.polys.numberfields.minpoly import _choose_factor, minimal_polynomial
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly, PurePoly, factor_list
from sympy.utilities import public
from mpmath import MPContext
@public
def primitive_element(extension, x=None, *, ex=False, polys=False):
    """
    Find a single generator for a number field given by several generators.

    Explanation
    ===========

    The basic problem is this: Given several algebraic numbers
    $\\alpha_1, \\alpha_2, \\ldots, \\alpha_n$, find a single algebraic number
    $\\theta$ such that
    $\\mathbb{Q}(\\alpha_1, \\alpha_2, \\ldots, \\alpha_n) = \\mathbb{Q}(\\theta)$.

    This function actually guarantees that $\\theta$ will be a linear
    combination of the $\\alpha_i$, with non-negative integer coefficients.

    Furthermore, if desired, this function will tell you how to express each
    $\\alpha_i$ as a $\\mathbb{Q}$-linear combination of the powers of $\\theta$.

    Examples
    ========

    >>> from sympy import primitive_element, sqrt, S, minpoly, simplify
    >>> from sympy.abc import x
    >>> f, lincomb, reps = primitive_element([sqrt(2), sqrt(3)], x, ex=True)

    Then ``lincomb`` tells us the primitive element as a linear combination of
    the given generators ``sqrt(2)`` and ``sqrt(3)``.

    >>> print(lincomb)
    [1, 1]

    This means the primtiive element is $\\sqrt{2} + \\sqrt{3}$.
    Meanwhile ``f`` is the minimal polynomial for this primitive element.

    >>> print(f)
    x**4 - 10*x**2 + 1
    >>> print(minpoly(sqrt(2) + sqrt(3), x))
    x**4 - 10*x**2 + 1

    Finally, ``reps`` (which was returned only because we set keyword arg
    ``ex=True``) tells us how to recover each of the generators $\\sqrt{2}$ and
    $\\sqrt{3}$ as $\\mathbb{Q}$-linear combinations of the powers of the
    primitive element $\\sqrt{2} + \\sqrt{3}$.

    >>> print([S(r) for r in reps[0]])
    [1/2, 0, -9/2, 0]
    >>> theta = sqrt(2) + sqrt(3)
    >>> print(simplify(theta**3/2 - 9*theta/2))
    sqrt(2)
    >>> print([S(r) for r in reps[1]])
    [-1/2, 0, 11/2, 0]
    >>> print(simplify(-theta**3/2 + 11*theta/2))
    sqrt(3)

    Parameters
    ==========

    extension : list of :py:class:`~.Expr`
        Each expression must represent an algebraic number $\\alpha_i$.
    x : :py:class:`~.Symbol`, optional (default=None)
        The desired symbol to appear in the computed minimal polynomial for the
        primitive element $\\theta$. If ``None``, we use a dummy symbol.
    ex : boolean, optional (default=False)
        If and only if ``True``, compute the representation of each $\\alpha_i$
        as a $\\mathbb{Q}$-linear combination over the powers of $\\theta$.
    polys : boolean, optional (default=False)
        If ``True``, return the minimal polynomial as a :py:class:`~.Poly`.
        Otherwise return it as an :py:class:`~.Expr`.

    Returns
    =======

    Pair (f, coeffs) or triple (f, coeffs, reps), where:
        ``f`` is the minimal polynomial for the primitive element.
        ``coeffs`` gives the primitive element as a linear combination of the
        given generators.
        ``reps`` is present if and only if argument ``ex=True`` was passed,
        and is a list of lists of rational numbers. Each list gives the
        coefficients of falling powers of the primitive element, to recover
        one of the original, given generators.

    """
    if not extension:
        raise ValueError('Cannot compute primitive element for empty extension')
    extension = [_sympify(ext) for ext in extension]
    if x is not None:
        x, cls = (sympify(x), Poly)
    else:
        x, cls = (Dummy('x'), PurePoly)
    if not ex:
        gen, coeffs = (extension[0], [1])
        g = minimal_polynomial(gen, x, polys=True)
        for ext in extension[1:]:
            if ext.is_Rational:
                coeffs.append(0)
                continue
            _, factors = factor_list(g, extension=ext)
            g = _choose_factor(factors, x, gen)
            s, _, g = g.sqf_norm()
            gen += s * ext
            coeffs.append(s)
        if not polys:
            return (g.as_expr(), coeffs)
        else:
            return (cls(g), coeffs)
    gen, coeffs = (extension[0], [1])
    f = minimal_polynomial(gen, x, polys=True)
    K = QQ.algebraic_field((f, gen))
    reps = [K.unit]
    for ext in extension[1:]:
        if ext.is_Rational:
            coeffs.append(0)
            reps.append(K.convert(ext))
            continue
        p = minimal_polynomial(ext, x, polys=True)
        L = QQ.algebraic_field((p, ext))
        _, factors = factor_list(f, domain=L)
        f = _choose_factor(factors, x, gen)
        s, g, f = f.sqf_norm()
        gen += s * ext
        coeffs.append(s)
        K = QQ.algebraic_field((f, gen))
        h = _switch_domain(g, K)
        erep = _linsolve(h.gcd(p))
        ogen = K.unit - s * erep
        reps = [dup_eval(_.rep, ogen, K) for _ in reps] + [erep]
    if K.ext.root.is_Rational:
        H = [K.convert(_).rep for _ in extension]
        coeffs = [0] * len(extension)
        f = cls(x, domain=QQ)
    else:
        H = [_.rep for _ in reps]
    if not polys:
        return (f.as_expr(), coeffs, H)
    else:
        return (f, coeffs, H)
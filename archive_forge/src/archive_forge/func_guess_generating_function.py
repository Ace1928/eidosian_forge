from sympy.concrete.products import (Product, product)
from sympy.core import Function, S
from sympy.core.add import Add
from sympy.core.numbers import Integer, Rational
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import integrate
from sympy.polys.polyfuncs import rational_interpolate as rinterp
from sympy.polys.polytools import lcm
from sympy.simplify.radsimp import denom
from sympy.utilities import public
@public
def guess_generating_function(v, X=Symbol('x'), types=['all'], maxsqrtn=2):
    """
    Tries to "guess" a generating function for a sequence of rational numbers v.
    Only a few patterns are implemented yet.

    Explanation
    ===========

    The function returns a dictionary where keys are the name of a given type of
    generating function. Six types are currently implemented:

         type  |  formal definition
        -------+----------------------------------------------------------------
        ogf    | f(x) = Sum(            a_k * x^k       ,  k: 0..infinity )
        egf    | f(x) = Sum(            a_k * x^k / k!  ,  k: 0..infinity )
        lgf    | f(x) = Sum( (-1)^(k+1) a_k * x^k / k   ,  k: 1..infinity )
               |        (with initial index being hold as 1 rather than 0)
        hlgf   | f(x) = Sum(            a_k * x^k / k   ,  k: 1..infinity )
               |        (with initial index being hold as 1 rather than 0)
        lgdogf | f(x) = derivate( log(Sum( a_k * x^k, k: 0..infinity )), x)
        lgdegf | f(x) = derivate( log(Sum( a_k * x^k / k!, k: 0..infinity )), x)

    In order to spare time, the user can select only some types of generating
    functions (default being ['all']). While forgetting to use a list in the
    case of a single type may seem to work most of the time as in: types='ogf'
    this (convenient) syntax may lead to unexpected extra results in some cases.

    Discarding a type when calling the function does not mean that the type will
    not be present in the returned dictionary; it only means that no extra
    computation will be performed for that type, but the function may still add
    it in the result when it can be easily converted from another type.

    Two generating functions (lgdogf and lgdegf) are not even computed if the
    initial term of the sequence is 0; it may be useful in that case to try
    again after having removed the leading zeros.

    Examples
    ========

    >>> from sympy.concrete.guess import guess_generating_function as ggf
    >>> ggf([k+1 for k in range(12)], types=['ogf', 'lgf', 'hlgf'])
    {'hlgf': 1/(1 - x), 'lgf': 1/(x + 1), 'ogf': 1/(x**2 - 2*x + 1)}

    >>> from sympy import sympify
    >>> l = sympify("[3/2, 11/2, 0, -121/2, -363/2, 121]")
    >>> ggf(l)
    {'ogf': (x + 3/2)/(11*x**2 - 3*x + 1)}

    >>> from sympy import fibonacci
    >>> ggf([fibonacci(k) for k in range(5, 15)], types=['ogf'])
    {'ogf': (3*x + 5)/(-x**2 - x + 1)}

    >>> from sympy import factorial
    >>> ggf([factorial(k) for k in range(12)], types=['ogf', 'egf', 'lgf'])
    {'egf': 1/(1 - x)}

    >>> ggf([k+1 for k in range(12)], types=['egf'])
    {'egf': (x + 1)*exp(x), 'lgdegf': (x + 2)/(x + 1)}

    N-th root of a rational function can also be detected (below is an example
    coming from the sequence A108626 from https://oeis.org).
    The greatest n-th root to be tested is specified as maxsqrtn (default 2).

    >>> ggf([1, 2, 5, 14, 41, 124, 383, 1200, 3799, 12122, 38919])['ogf']
    sqrt(1/(x**4 + 2*x**2 - 4*x + 1))

    References
    ==========

    .. [1] "Concrete Mathematics", R.L. Graham, D.E. Knuth, O. Patashnik
    .. [2] https://oeis.org/wiki/Generating_functions

    """
    if 'all' in types:
        types = ('ogf', 'egf', 'lgf', 'hlgf', 'lgdogf', 'lgdegf')
    result = {}
    if 'ogf' in types:
        t = [1] + [0] * (len(v) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * v[i] for i in range(n + 1))) for n in range(len(v))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['ogf'] = g ** Rational(1, d + 1)
                break
    if 'egf' in types:
        w, f = ([], S.One)
        for i, k in enumerate(v):
            f *= i if i else 1
            w.append(k / f)
        t = [1] + [0] * (len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * w[i] for i in range(n + 1))) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['egf'] = g ** Rational(1, d + 1)
                break
    if 'lgf' in types:
        w, f = ([], S.NegativeOne)
        for i, k in enumerate(v):
            f = -f
            w.append(f * k / Integer(i + 1))
        t = [1] + [0] * (len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * w[i] for i in range(n + 1))) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgf'] = g ** Rational(1, d + 1)
                break
    if 'hlgf' in types:
        w = []
        for i, k in enumerate(v):
            w.append(k / Integer(i + 1))
        t = [1] + [0] * (len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * w[i] for i in range(n + 1))) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['hlgf'] = g ** Rational(1, d + 1)
                break
    if v[0] != 0 and ('lgdogf' in types or ('ogf' in types and 'ogf' not in result)):
        a, w = (sympify(v[0]), [])
        for n in range(len(v) - 1):
            w.append((v[n + 1] * (n + 1) - sum((w[-i - 1] * v[i + 1] for i in range(n)))) / a)
        t = [1] + [0] * (len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * w[i] for i in range(n + 1))) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgdogf'] = g ** Rational(1, d + 1)
                if 'ogf' not in result:
                    result['ogf'] = exp(integrate(result['lgdogf'], X))
                break
    if v[0] != 0 and ('lgdegf' in types or ('egf' in types and 'egf' not in result)):
        z, f = ([], S.One)
        for i, k in enumerate(v):
            f *= i if i else 1
            z.append(k / f)
        a, w = (z[0], [])
        for n in range(len(z) - 1):
            w.append((z[n + 1] * (n + 1) - sum((w[-i - 1] * z[i + 1] for i in range(n)))) / a)
        t = [1] + [0] * (len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * w[i] for i in range(n + 1))) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgdegf'] = g ** Rational(1, d + 1)
                if 'egf' not in result:
                    result['egf'] = exp(integrate(result['lgdegf'], X))
                break
    return result
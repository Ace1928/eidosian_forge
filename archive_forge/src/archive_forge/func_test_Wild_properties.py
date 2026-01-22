import threading
from sympy.core.function import Function, UndefinedFunction
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (GreaterThan, LessThan, StrictGreaterThan, StrictLessThan)
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify  # can't import as S yet
from sympy.core.symbol import uniquely_named_symbol, _symbol, Str
from sympy.testing.pytest import raises, skip_under_pyodide
from sympy.core.symbol import disambiguate
def test_Wild_properties():
    S = sympify
    x = Symbol('x')
    y = Symbol('y')
    p = Symbol('p', positive=True)
    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True, positive=True)
    given_patterns = [x, y, p, k, -k, n, -n, S(-3), S(3), pi, Rational(3, 2), I]
    integerp = lambda k: k.is_integer
    positivep = lambda k: k.is_positive
    symbolp = lambda k: k.is_Symbol
    realp = lambda k: k.is_extended_real
    S = Wild('S', properties=[symbolp])
    R = Wild('R', properties=[realp])
    Y = Wild('Y', exclude=[x, p, k, n])
    P = Wild('P', properties=[positivep])
    K = Wild('K', properties=[integerp])
    N = Wild('N', properties=[positivep, integerp])
    given_wildcards = [S, R, Y, P, K, N]
    goodmatch = {S: (x, y, p, k, n), R: (p, k, -k, n, -n, -3, 3, pi, Rational(3, 2)), Y: (y, -3, 3, pi, Rational(3, 2), I), P: (p, n, 3, pi, Rational(3, 2)), K: (k, -k, n, -n, -3, 3), N: (n, 3)}
    for A in given_wildcards:
        for pat in given_patterns:
            d = pat.match(A)
            if pat in goodmatch[A]:
                assert d[A] in goodmatch[A]
            else:
                assert d is None
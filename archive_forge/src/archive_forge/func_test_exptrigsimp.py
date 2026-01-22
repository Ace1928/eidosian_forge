from itertools import product
from sympy.core.function import (Subs, count_ops, diff, expand)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (asec, acsc)
from sympy.functions.elementary.trigonometric import (acot, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import (exptrigsimp, trigsimp)
from sympy.testing.pytest import XFAIL
from sympy.abc import x, y
def test_exptrigsimp():

    def valid(a, b):
        from sympy.core.random import verify_numerically as tn
        if not (tn(a, b) and a == b):
            return False
        return True
    assert exptrigsimp(exp(x) + exp(-x)) == 2 * cosh(x)
    assert exptrigsimp(exp(x) - exp(-x)) == 2 * sinh(x)
    assert exptrigsimp((2 * exp(x) - 2 * exp(-x)) / (exp(x) + exp(-x))) == 2 * tanh(x)
    assert exptrigsimp((2 * exp(2 * x) - 2) / (exp(2 * x) + 1)) == 2 * tanh(x)
    e = [cos(x) + I * sin(x), cos(x) - I * sin(x), cosh(x) - sinh(x), cosh(x) + sinh(x)]
    ok = [exp(I * x), exp(-I * x), exp(-x), exp(x)]
    assert all((valid(i, j) for i, j in zip([exptrigsimp(ei) for ei in e], ok)))
    ue = [cos(x) + sin(x), cos(x) - sin(x), cosh(x) + I * sinh(x), cosh(x) - I * sinh(x)]
    assert [exptrigsimp(ei) == ei for ei in ue]
    res = []
    ok = [y * tanh(1), 1 / (y * tanh(1)), I * y * tan(1), -I / (y * tan(1)), y * tanh(x), 1 / (y * tanh(x)), I * y * tan(x), -I / (y * tan(x)), y * tanh(1 + I), 1 / (y * tanh(1 + I))]
    for a in (1, I, x, I * x, 1 + I):
        w = exp(a)
        eq = y * (w - 1 / w) / (w + 1 / w)
        res.append(simplify(eq))
        res.append(simplify(1 / eq))
    assert all((valid(i, j) for i, j in zip(res, ok)))
    for a in range(1, 3):
        w = exp(a)
        e = w + 1 / w
        s = simplify(e)
        assert s == exptrigsimp(e)
        assert valid(s, 2 * cosh(a))
        e = w - 1 / w
        s = simplify(e)
        assert s == exptrigsimp(e)
        assert valid(s, 2 * sinh(a))
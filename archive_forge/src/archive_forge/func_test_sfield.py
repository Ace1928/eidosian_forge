from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_sfield():
    x = symbols('x')
    F = FracField((E, exp(exp(x)), exp(x)), ZZ, lex)
    e, exex, ex = F.gens
    assert sfield(exp(x) * exp(exp(x) + 1 + log(exp(x) + 3) / 2) ** 2 / (exp(x) + 3)) == (F, e ** 2 * exex ** 2 * ex)
    F = FracField((x, exp(1 / x), log(x), x ** QQ(1, 3)), ZZ, lex)
    _, ex, lg, x3 = F.gens
    assert sfield(((x - 3) * log(x) + 4 * x ** 2) * exp(1 / x + log(x) / 3) / x ** 2) == (F, (4 * F.x ** 2 * ex + F.x * ex * lg - 3 * ex * lg) / x3 ** 5)
    F = FracField((x, log(x), sqrt(x + log(x))), ZZ, lex)
    _, lg, srt = F.gens
    assert sfield((x + 1) / (x * (x + log(x)) ** QQ(3, 2)) - 1 / (x * log(x) ** 2)) == (F, (F.x * lg ** 2 - F.x * srt + lg ** 2 - lg * srt) / (F.x ** 2 * lg ** 2 * srt + F.x * lg ** 3 * srt))
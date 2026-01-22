import pickle
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar, Replacer
def test_matchpy_optional():
    if matchpy is None:
        skip('matchpy not installed')
    from matchpy import Pattern, Substitution
    from matchpy import ManyToOneReplacer, ReplacementRule
    p = WildDot('p', optional=1)
    q = WildDot('q', optional=0)
    pattern = p * x + q
    expr1 = 2 * x
    pa, subst = _get_first_match(expr1, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': 2, 'q': 0})
    expr2 = x + 3
    pa, subst = _get_first_match(expr2, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': 1, 'q': 3})
    expr3 = x
    pa, subst = _get_first_match(expr3, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': 1, 'q': 0})
    expr4 = x * y + z
    pa, subst = _get_first_match(expr4, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': y, 'q': z})
    replacer = ManyToOneReplacer()
    replacer.add(ReplacementRule(Pattern(pattern), lambda p, q: sin(p) * cos(q)))
    assert replacer.replace(expr1) == sin(2) * cos(0)
    assert replacer.replace(expr2) == sin(1) * cos(3)
    assert replacer.replace(expr3) == sin(1) * cos(0)
    assert replacer.replace(expr4) == sin(y) * cos(z)
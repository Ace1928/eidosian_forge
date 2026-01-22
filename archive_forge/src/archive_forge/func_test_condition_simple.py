from sympy.unify.rewrite import rewriterule
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x, y
from sympy.strategies.rl import rebuild
from sympy.assumptions import Q
def test_condition_simple():
    rl = rewriterule(x, x + 1, [x], lambda x: x < 10)
    assert not list(rl(S(15)))
    assert rebuild(next(rl(S(5)))) == 6
from sympy.unify.rewrite import rewriterule
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x, y
from sympy.strategies.rl import rebuild
from sympy.assumptions import Q
def test_condition_multiple():
    rl = rewriterule(x + y, x ** y, [x, y], lambda x, y: x.is_integer)
    a = Symbol('a')
    b = Symbol('b', integer=True)
    expr = a + b
    assert list(rl(expr)) == [b ** a]
    c = Symbol('c', integer=True)
    d = Symbol('d', integer=True)
    assert set(rl(c + d)) == {c ** d, d ** c}
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.strategies.branch.traverse import top_down, sall
from sympy.strategies.branch.core import do_one, identity
def test_top_down_easy():
    expr = Basic(S(1), S(2))
    expected = Basic(S(2), S(3))
    brl = top_down(inc)
    assert set(brl(expr)) == {expected}
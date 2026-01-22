from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.strategies.branch.traverse import top_down, sall
from sympy.strategies.branch.core import do_one, identity
def test_top_down_harder_function():

    def split5(x):
        if x == 5:
            yield (x - 1)
            yield (x + 1)
    expr = Basic(Basic(S(5), S(6)), S(1))
    expected = {Basic(Basic(S(4), S(6)), S(1)), Basic(Basic(S(6), S(6)), S(1))}
    brl = top_down(split5)
    assert set(brl(expr)) == expected
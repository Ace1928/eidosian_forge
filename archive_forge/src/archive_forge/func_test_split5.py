from sympy.strategies.branch.tools import canon
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
def test_split5():
    expr = Basic(S(2), Basic(S(5), S(3)), S(8))
    expected = {Basic(S(0), Basic(S(0), S(0)), S(10)), Basic(S(0), Basic(S(10), S(0)), S(10))}
    brl = canon(branch5)
    assert set(brl(expr)) == expected
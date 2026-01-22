from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.operations import AssocOp, LatticeOp
from sympy.testing.pytest import raises
from sympy.core.sympify import SympifyError
from sympy.core.add import Add, add
from sympy.core.mul import Mul, mul
def test_AssocOp_flatten():
    a, b, c, d = symbols('a,b,c,d')

    class MyAssoc(AssocOp):
        identity = S.One
    assert MyAssoc(a, MyAssoc(b, c)).args == MyAssoc(MyAssoc(a, b), c).args == MyAssoc(MyAssoc(a, b, c)).args == MyAssoc(a, b, c).args == (a, b, c)
    u = MyAssoc(b, c)
    v = MyAssoc(u, d, evaluate=False)
    assert v.args == (u, d)
    assert MyAssoc(a, v).args == (a, b, c, d)
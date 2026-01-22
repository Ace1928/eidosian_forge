from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.operations import AssocOp, LatticeOp
from sympy.testing.pytest import raises
from sympy.core.sympify import SympifyError
from sympy.core.add import Add, add
from sympy.core.mul import Mul, mul
def test_issue_14025():
    a, b, c, d = symbols('a,b,c,d', commutative=False)
    assert Mul(a, b, c).has(c * b) == False
    assert Mul(a, b, c).has(b * c) == True
    assert Mul(a, b, c, d).has(b * c * d) == True
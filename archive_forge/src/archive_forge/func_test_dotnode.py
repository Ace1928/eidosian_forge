from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import (Float, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing.repr import srepr
from sympy.abc import x
def test_dotnode():
    assert dotnode(x, repeat=False) == '"Symbol(\'x\')" ["color"="black", "label"="x", "shape"="ellipse"];'
    assert dotnode(x + 2, repeat=False) == '"Add(Integer(2), Symbol(\'x\'))" ["color"="black", "label"="Add", "shape"="ellipse"];', dotnode(x + 2, repeat=0)
    assert dotnode(x + x ** 2, repeat=False) == '"Add(Symbol(\'x\'), Pow(Symbol(\'x\'), Integer(2)))" ["color"="black", "label"="Add", "shape"="ellipse"];'
    assert dotnode(x + x ** 2, repeat=True) == '"Add(Symbol(\'x\'), Pow(Symbol(\'x\'), Integer(2)))_()" ["color"="black", "label"="Add", "shape"="ellipse"];'
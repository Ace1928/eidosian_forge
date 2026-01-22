from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import (Float, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing.repr import srepr
from sympy.abc import x
def test_dotprint_depth():
    text = dotprint(3 * x + 2, depth=1)
    assert dotnode(3 * x + 2) in text
    assert dotnode(x) not in text
    text = dotprint(3 * x + 2)
    assert 'depth' not in text
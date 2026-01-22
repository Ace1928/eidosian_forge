from sympy.core.symbol import symbols
from sympy.printing.codeprinter import ccode
from sympy.codegen.ast import Declaration, Variable, float64, int64, String, CodeBlock
from sympy.codegen.cnodes import (
def test_goto_Label():
    s = 'early_exit'
    g = goto(s)
    assert g.func(*g.args) == g
    assert g != goto('foobar')
    assert ccode(g) == 'goto early_exit'
    l1 = Label(s)
    assert ccode(l1) == 'early_exit:'
    assert l1 == Label('early_exit')
    assert l1 != Label('foobar')
    body = [PreIncrement(x)]
    l2 = Label(s, body)
    assert l2.name == String('early_exit')
    assert l2.body == CodeBlock(PreIncrement(x))
    assert ccode(l2) == 'early_exit:\n++(x);'
    body = [PreIncrement(x), PreDecrement(y)]
    l2 = Label(s, body)
    assert l2.name == String('early_exit')
    assert l2.body == CodeBlock(PreIncrement(x), PreDecrement(y))
    assert ccode(l2) == 'early_exit:\n{\n   ++(x);\n   --(y);\n}'
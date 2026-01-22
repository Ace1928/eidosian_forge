from sympy.core.symbol import symbols
from sympy.printing.codeprinter import ccode
from sympy.codegen.ast import Declaration, Variable, float64, int64, String, CodeBlock
from sympy.codegen.cnodes import (
def test_PostDecrement():
    p = PostDecrement(x)
    assert p.func(*p.args) == p
    assert ccode(p) == '(x)--'
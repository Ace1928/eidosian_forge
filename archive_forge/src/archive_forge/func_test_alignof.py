from sympy.core.symbol import symbols
from sympy.printing.codeprinter import ccode
from sympy.codegen.ast import Declaration, Variable, float64, int64, String, CodeBlock
from sympy.codegen.cnodes import (
def test_alignof():
    ax = alignof(x)
    assert ccode(ax) == 'alignof(x)'
    assert ax.func(*ax.args) == ax
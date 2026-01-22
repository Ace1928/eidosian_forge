import math
from sympy.core.containers import Tuple
from sympy.core.numbers import nan, oo, Float, Integer
from sympy.core.relational import Lt
from sympy.core.symbol import symbols, Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.testing.pytest import raises
from sympy.codegen.ast import (
def test_CodeBlock_cse():
    c1 = CodeBlock(Assignment(y, 1), Assignment(x, sin(y)), Assignment(z, sin(y)), Assignment(t, x * z))
    assert c1.cse() == CodeBlock(Assignment(y, 1), Assignment(x0, sin(y)), Assignment(x, x0), Assignment(z, x0), Assignment(t, x * z))
    raises(NotImplementedError, lambda: CodeBlock(Assignment(x, 1), Assignment(y, 1), Assignment(y, 2)).cse())
    c2 = CodeBlock(Assignment(x0, sin(y) + 1), Assignment(x1, 2 * sin(y)), Assignment(z, x * y))
    assert c2.cse() == CodeBlock(Assignment(x2, sin(y)), Assignment(x0, x2 + 1), Assignment(x1, 2 * x2), Assignment(z, x * y))
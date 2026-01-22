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
def test_Type__cast_check__complex_floating_point():
    val9_11 = 123.456789049 + 0.123456789049j
    raises(ValueError, lambda: c64.cast_check(0.12345678949 + 0.12345678949j))
    assert abs(val9_11 - c64.cast_check(val9_11) - 4.9e-08) < 1e-08
    dcm21 = Float('0.123456789012345670499') + 1e-20j
    assert abs(dcm21 - c128.cast_check(dcm21) - 4.99e-19) < 1e-19
    v19 = Float('0.1234567890123456749') + 1j * Float('0.1234567890123456749')
    raises(ValueError, lambda: c128.cast_check(v19))
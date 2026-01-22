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
def test_Type():
    t = Type('MyType')
    assert len(t.args) == 1
    assert t.name == String('MyType')
    assert str(t) == 'MyType'
    assert repr(t) == "Type(String('MyType'))"
    assert Type(t) == t
    assert t.func(*t.args) == t
    t1 = Type('t1')
    t2 = Type('t2')
    assert t1 != t2
    assert t1 == t1 and t2 == t2
    t1b = Type('t1')
    assert t1 == t1b
    assert t2 != t1b
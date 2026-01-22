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
def test_Pointer():
    p = Pointer(x)
    assert p.symbol == x
    assert p.type == untyped
    assert value_const not in p.attrs
    assert pointer_const not in p.attrs
    assert p.func(*p.args) == p
    u = symbols('u', real=True)
    pu = Pointer(u, type=Type.from_expr(u), attrs={value_const, pointer_const})
    assert pu.symbol is u
    assert pu.type == real
    assert value_const in pu.attrs
    assert pointer_const in pu.attrs
    assert pu.func(*pu.args) == pu
    i = symbols('i', integer=True)
    deref = pu[i]
    assert deref.indices == (i,)
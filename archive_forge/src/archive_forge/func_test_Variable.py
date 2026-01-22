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
def test_Variable():
    v = Variable(x, type=real)
    assert v == Variable(v)
    assert v == Variable('x', type=real)
    assert v.symbol == x
    assert v.type == real
    assert value_const not in v.attrs
    assert v.func(*v.args) == v
    assert str(v) == 'Variable(x, type=real)'
    w = Variable(y, f32, attrs={value_const})
    assert w.symbol == y
    assert w.type == f32
    assert value_const in w.attrs
    assert w.func(*w.args) == w
    v_n = Variable(n, type=Type.from_expr(n))
    assert v_n.type == integer
    assert v_n.func(*v_n.args) == v_n
    v_i = Variable(i, type=Type.from_expr(n))
    assert v_i.type == integer
    assert v_i != v_n
    a_i = Variable.deduced(i)
    assert a_i.type == integer
    assert Variable.deduced(Symbol('x', real=True)).type == real
    assert a_i.func(*a_i.args) == a_i
    v_n2 = Variable.deduced(n, value=3.5, cast_check=False)
    assert v_n2.func(*v_n2.args) == v_n2
    assert abs(v_n2.value - 3.5) < 1e-15
    raises(ValueError, lambda: Variable.deduced(n, value=3.5, cast_check=True))
    v_n3 = Variable.deduced(n)
    assert v_n3.type == integer
    assert str(v_n3) == 'Variable(n, type=integer)'
    assert Variable.deduced(z, value=3).type == integer
    assert Variable.deduced(z, value=3.0).type == real
    assert Variable.deduced(z, value=3.0 + 1j).type == complex_
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
def test_String():
    st = String('foobar')
    assert st.is_Atom
    assert st == String('foobar')
    assert st.text == 'foobar'
    assert st.func(**st.kwargs()) == st
    assert st.func(*st.args) == st

    class Signifier(String):
        pass
    si = Signifier('foobar')
    assert si != st
    assert si.text == st.text
    s = String('foo')
    assert str(s) == 'foo'
    assert repr(s) == "String('foo')"
import threading
from sympy.core.function import Function, UndefinedFunction
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (GreaterThan, LessThan, StrictGreaterThan, StrictLessThan)
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify  # can't import as S yet
from sympy.core.symbol import uniquely_named_symbol, _symbol, Str
from sympy.testing.pytest import raises, skip_under_pyodide
from sympy.core.symbol import disambiguate
def test_Dummy_force_dummy_index():
    raises(AssertionError, lambda: Dummy(dummy_index=1))
    assert Dummy('d', dummy_index=2) == Dummy('d', dummy_index=2)
    assert Dummy('d1', dummy_index=2) != Dummy('d2', dummy_index=2)
    d1 = Dummy('d', dummy_index=3)
    d2 = Dummy('d')
    assert d1 != d2
    d3 = Dummy('d', dummy_index=3)
    assert d1 == d3
    assert Dummy()._count == Dummy('d', dummy_index=3)._count
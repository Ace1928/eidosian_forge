import threading
from sympy.core.function import Function, UndefinedFunction
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (GreaterThan, LessThan, StrictGreaterThan, StrictLessThan)
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify  # can't import as S yet
from sympy.core.symbol import uniquely_named_symbol, _symbol, Str
from sympy.testing.pytest import raises, skip_under_pyodide
from sympy.core.symbol import disambiguate
def test_disambiguate():
    x, y, y_1, _x, x_1, x_2 = symbols('x y y_1 _x x_1 x_2')
    t1 = (Dummy('y'), _x, Dummy('x'), Dummy('x'))
    t2 = (Dummy('x'), Dummy('x'))
    t3 = (Dummy('x'), Dummy('y'))
    t4 = (x, Dummy('x'))
    t5 = (Symbol('x', integer=True), x, Symbol('x_1'))
    assert disambiguate(*t1) == (y, x_2, x, x_1)
    assert disambiguate(*t2) == (x, x_1)
    assert disambiguate(*t3) == (x, y)
    assert disambiguate(*t4) == (x_1, x)
    assert disambiguate(*t5) == (t5[0], x_2, x_1)
    assert disambiguate(*t5)[0] != x
    t6 = (_x, Dummy('x') / y)
    t7 = (y * Dummy('y'), y)
    assert disambiguate(*t6) == (x_1, x / y)
    assert disambiguate(*t7) == (y * y_1, y_1)
    assert disambiguate(Dummy('x_1'), Dummy('x_1')) == (x_1, Symbol('x_1_1'))
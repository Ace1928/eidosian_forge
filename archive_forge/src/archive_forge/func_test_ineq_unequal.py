import threading
from sympy.core.function import Function, UndefinedFunction
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (GreaterThan, LessThan, StrictGreaterThan, StrictLessThan)
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify  # can't import as S yet
from sympy.core.symbol import uniquely_named_symbol, _symbol, Str
from sympy.testing.pytest import raises, skip_under_pyodide
from sympy.core.symbol import disambiguate
def test_ineq_unequal():
    S = sympify
    x, y, z = symbols('x,y,z')
    e = (S(-1) >= x, S(-1) >= y, S(-1) >= z, S(-1) > x, S(-1) > y, S(-1) > z, S(-1) <= x, S(-1) <= y, S(-1) <= z, S(-1) < x, S(-1) < y, S(-1) < z, S(0) >= x, S(0) >= y, S(0) >= z, S(0) > x, S(0) > y, S(0) > z, S(0) <= x, S(0) <= y, S(0) <= z, S(0) < x, S(0) < y, S(0) < z, S('3/7') >= x, S('3/7') >= y, S('3/7') >= z, S('3/7') > x, S('3/7') > y, S('3/7') > z, S('3/7') <= x, S('3/7') <= y, S('3/7') <= z, S('3/7') < x, S('3/7') < y, S('3/7') < z, S(1.5) >= x, S(1.5) >= y, S(1.5) >= z, S(1.5) > x, S(1.5) > y, S(1.5) > z, S(1.5) <= x, S(1.5) <= y, S(1.5) <= z, S(1.5) < x, S(1.5) < y, S(1.5) < z, S(2) >= x, S(2) >= y, S(2) >= z, S(2) > x, S(2) > y, S(2) > z, S(2) <= x, S(2) <= y, S(2) <= z, S(2) < x, S(2) < y, S(2) < z, x >= -1, y >= -1, z >= -1, x > -1, y > -1, z > -1, x <= -1, y <= -1, z <= -1, x < -1, y < -1, z < -1, x >= 0, y >= 0, z >= 0, x > 0, y > 0, z > 0, x <= 0, y <= 0, z <= 0, x < 0, y < 0, z < 0, x >= 1.5, y >= 1.5, z >= 1.5, x > 1.5, y > 1.5, z > 1.5, x <= 1.5, y <= 1.5, z <= 1.5, x < 1.5, y < 1.5, z < 1.5, x >= 2, y >= 2, z >= 2, x > 2, y > 2, z > 2, x <= 2, y <= 2, z <= 2, x < 2, y < 2, z < 2, x >= y, x >= z, y >= x, y >= z, z >= x, z >= y, x > y, x > z, y > x, y > z, z > x, z > y, x <= y, x <= z, y <= x, y <= z, z <= x, z <= y, x < y, x < z, y < x, y < z, z < x, z < y, x - pi >= y + z, y - pi >= x + z, z - pi >= x + y, x - pi > y + z, y - pi > x + z, z - pi > x + y, x - pi <= y + z, y - pi <= x + z, z - pi <= x + y, x - pi < y + z, y - pi < x + z, z - pi < x + y, True, False)
    left_e = e[:-1]
    for i, e1 in enumerate(left_e):
        for e2 in e[i + 1:]:
            assert e1 != e2
import sympy
import tempfile
import os
from sympy.core.mod import Mod
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.external import import_module
from sympy.tensor import IndexedBase, Idx
from sympy.utilities.autowrap import autowrap, ufuncify, CodeWrapError
from sympy.testing.pytest import skip
def test_issue_15230():
    has_module('f2py')
    x, y = symbols('x, y')
    expr = Mod(x, 3.0) - Mod(y, -2.0)
    f = autowrap(expr, args=[x, y], language='F95')
    exp_res = float(expr.xreplace({x: 3.5, y: 2.7}).evalf())
    assert abs(f(3.5, 2.7) - exp_res) < 1e-14
    x, y = symbols('x, y', integer=True)
    expr = Mod(x, 3) - Mod(y, -2)
    f = autowrap(expr, args=[x, y], language='F95')
    assert f(3, 2) == expr.xreplace({x: 3, y: 2})
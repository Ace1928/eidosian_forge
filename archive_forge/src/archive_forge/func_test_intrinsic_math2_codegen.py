from io import StringIO
from sympy.core import symbols, Eq, pi, Catalan, Lambda, Dummy
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import (
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
def test_intrinsic_math2_codegen():
    from sympy.functions.elementary.trigonometric import atan2
    x, y = symbols('x,y')
    name_expr = [('test_atan2', atan2(x, y)), ('test_pow', x ** y)]
    result = codegen(name_expr, 'F95', 'file', header=False, empty=False)
    assert result[0][0] == 'file.f90'
    expected = 'REAL*8 function test_atan2(x, y)\nimplicit none\nREAL*8, intent(in) :: x\nREAL*8, intent(in) :: y\ntest_atan2 = atan2(x, y)\nend function\nREAL*8 function test_pow(x, y)\nimplicit none\nREAL*8, intent(in) :: x\nREAL*8, intent(in) :: y\ntest_pow = x**y\nend function\n'
    assert result[0][1] == expected
    assert result[1][0] == 'file.h'
    expected = 'interface\nREAL*8 function test_atan2(x, y)\nimplicit none\nREAL*8, intent(in) :: x\nREAL*8, intent(in) :: y\nend function\nend interface\ninterface\nREAL*8 function test_pow(x, y)\nimplicit none\nREAL*8, intent(in) :: x\nREAL*8, intent(in) :: y\nend function\nend interface\n'
    assert result[1][1] == expected
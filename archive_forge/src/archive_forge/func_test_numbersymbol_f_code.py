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
def test_numbersymbol_f_code():
    routine = make_routine('test', pi ** Catalan)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = 'REAL*8 function test()\nimplicit none\nREAL*8, parameter :: Catalan = %sd0\nREAL*8, parameter :: pi = %sd0\ntest = pi**Catalan\nend function\n' % (Catalan.evalf(17), pi.evalf(17))
    assert source == expected
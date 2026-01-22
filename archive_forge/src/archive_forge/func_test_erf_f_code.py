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
def test_erf_f_code():
    x = symbols('x')
    routine = make_routine('test', erf(x) - erf(-2 * x))
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = 'REAL*8 function test(x)\nimplicit none\nREAL*8, intent(in) :: x\ntest = erf(x) + erf(2.0d0*x)\nend function\n'
    assert source == expected, source
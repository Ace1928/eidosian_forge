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
def test_fcode_matrixsymbol_slice_autoname():
    A = MatrixSymbol('A', 2, 3)
    name_expr = ('test', A[:, 1])
    result = codegen(name_expr, 'f95', 'test', header=False, empty=False)
    source = result[0][1]
    expected = 'subroutine test(A, out_%(hash)s)\nimplicit none\nREAL*8, intent(in), dimension(1:2, 1:3) :: A\nREAL*8, intent(out), dimension(1:2, 1:1) :: out_%(hash)s\nout_%(hash)s(1, 1) = A(1, 2)\nout_%(hash)s(2, 1) = A(2, 2)\nend subroutine\n'
    a = source.splitlines()[3]
    b = a.split('_')
    out = b[1]
    expected = expected % {'hash': out}
    assert source == expected
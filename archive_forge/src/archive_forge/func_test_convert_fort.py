from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises
from sympy.external import import_module
def test_convert_fort():
    src1 = src + '            a = b + c\n            s = p * q / r\n            '
    expr1.convert_to_expr(src1, 'f')
    exp_fort = expr1.convert_to_fortran()
    assert exp_fort == ['      integer*4 a', '      integer*4 b', '      integer*4 c', '      integer*4 d', '      real*8 p', '      real*8 q', '      real*8 r', '      real*8 s', '      a = b + c', '      s = p*q/r']
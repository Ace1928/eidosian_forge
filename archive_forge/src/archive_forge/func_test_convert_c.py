from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises
from sympy.external import import_module
def test_convert_c():
    src1 = src + '            a = b + c\n            s = p * q / r\n            '
    expr1.convert_to_expr(src1, 'f')
    exp_c = expr1.convert_to_c()
    assert exp_c == ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0', 'a = b + c;', 's = p*q/r;']
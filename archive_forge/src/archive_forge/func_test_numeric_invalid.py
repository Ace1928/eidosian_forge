import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_numeric_invalid(self):
    m = ConcreteModel()
    m.Y1 = BooleanVar()
    m.Y2 = BooleanVar()
    m.Y3 = BooleanVar()

    def iadd():
        m.Y3 += 2

    def isub():
        m.Y3 -= 2

    def imul():
        m.Y3 *= 2

    def idiv():
        m.Y3 /= 2

    def ipow():
        m.Y3 **= 2

    def iand():
        m.Y3 &= 2

    def ior():
        m.Y3 |= 2

    def ixor():
        m.Y3 ^= 2

    def invalid_expression_generator():
        yield (lambda: m.Y1 + m.Y2)
        yield (lambda: m.Y1 - m.Y2)
        yield (lambda: m.Y1 * m.Y2)
        yield (lambda: m.Y1 / m.Y2)
        yield (lambda: m.Y1 ** m.Y2)
        yield (lambda: m.Y1.land(0))
        yield (lambda: m.Y1.lor(0))
        yield (lambda: m.Y1.xor(0))
        yield (lambda: m.Y1.equivalent_to(0))
        yield (lambda: m.Y1.implies(0))
        yield (lambda: 0 + m.Y2)
        yield (lambda: 0 - m.Y2)
        yield (lambda: 0 * m.Y2)
        yield (lambda: 0 / m.Y2)
        yield (lambda: 0 ** m.Y2)
        yield (lambda: 0 & m.Y2)
        yield (lambda: 0 | m.Y2)
        yield (lambda: 0 ^ m.Y2)
        yield (lambda: m.Y3 + 2)
        yield (lambda: m.Y3 - 2)
        yield (lambda: m.Y3 * 2)
        yield (lambda: m.Y3 / 2)
        yield (lambda: m.Y3 ** 2)
        yield (lambda: m.Y3 & 2)
        yield (lambda: m.Y3 | 2)
        yield (lambda: m.Y3 ^ 2)
        yield iadd
        yield isub
        yield imul
        yield idiv
        yield ipow
        yield iand
        yield ior
        yield ixor
    numeric_error_msg = '(?:(?:unsupported operand type)|(?:operands do not support))'
    for i, invalid_expr_fcn in enumerate(invalid_expression_generator()):
        with self.assertRaisesRegex(TypeError, numeric_error_msg):
            _ = invalid_expr_fcn()

    def invalid_unary_expression_generator():
        yield (lambda: -m.Y1)
        yield (lambda: +m.Y1)
    for invalid_expr_fcn in invalid_unary_expression_generator():
        with self.assertRaisesRegex(TypeError, '(?:(?:bad operand type for unary)|(?:unsupported operand type for unary))'):
            _ = invalid_expr_fcn()

    def invalid_comparison_generator():
        yield (lambda: m.Y1 >= 0)
        yield (lambda: m.Y1 <= 0)
        yield (lambda: m.Y1 > 0)
        yield (lambda: m.Y1 < 0)
    comparison_error_msg = '(?:(?:unorderable types)|(?:not supported between instances of))'
    for invalid_expr_fcn in invalid_comparison_generator():
        with self.assertRaisesRegex(TypeError, comparison_error_msg):
            _ = invalid_expr_fcn()
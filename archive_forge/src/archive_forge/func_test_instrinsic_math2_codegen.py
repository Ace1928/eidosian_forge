from sympy.abc import x, y, z
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.codegen import codegen, make_routine, get_code_generator
import sys
import os
import tempfile
import subprocess
def test_instrinsic_math2_codegen():
    from sympy.core.evalf import N
    from sympy.functions.elementary.trigonometric import atan2
    name_expr = [('test_atan2', atan2(x, y)), ('test_pow', x ** y)]
    numerical_tests = []
    for name, expr in name_expr:
        for xval, yval in ((0.2, 1.3), (0.5, -0.2), (0.8, 0.8)):
            expected = N(expr.subs(x, xval).subs(y, yval))
            numerical_tests.append((name, (xval, yval), expected, 1e-14))
    for lang, commands in valid_lang_commands:
        run_test('intrinsic_math2', name_expr, numerical_tests, lang, commands)
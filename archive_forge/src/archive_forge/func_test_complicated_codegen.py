from sympy.abc import x, y, z
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.codegen import codegen, make_routine, get_code_generator
import sys
import os
import tempfile
import subprocess
def test_complicated_codegen():
    from sympy.core.evalf import N
    from sympy.functions.elementary.trigonometric import cos, sin, tan
    name_expr = [('test1', ((sin(x) + cos(y) + tan(z)) ** 7).expand()), ('test2', cos(cos(cos(cos(cos(cos(cos(cos(x + y + z)))))))))]
    numerical_tests = []
    for name, expr in name_expr:
        for xval, yval, zval in ((0.2, 1.3, -0.3), (0.5, -0.2, 0.0), (0.8, 2.1, 0.8)):
            expected = N(expr.subs(x, xval).subs(y, yval).subs(z, zval))
            numerical_tests.append((name, (xval, yval, zval), expected, 1e-12))
    for lang, commands in valid_lang_commands:
        run_test('complicated_codegen', name_expr, numerical_tests, lang, commands)
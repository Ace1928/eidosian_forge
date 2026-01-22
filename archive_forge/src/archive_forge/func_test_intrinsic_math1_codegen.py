from sympy.abc import x, y, z
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.codegen import codegen, make_routine, get_code_generator
import sys
import os
import tempfile
import subprocess
def test_intrinsic_math1_codegen():
    from sympy.core.evalf import N
    from sympy.functions import ln
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
    from sympy.functions.elementary.integers import ceiling, floor
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import acos, asin, atan, cos, sin, tan
    name_expr = [('test_fabs', abs(x)), ('test_acos', acos(x)), ('test_asin', asin(x)), ('test_atan', atan(x)), ('test_cos', cos(x)), ('test_cosh', cosh(x)), ('test_log', log(x)), ('test_ln', ln(x)), ('test_sin', sin(x)), ('test_sinh', sinh(x)), ('test_sqrt', sqrt(x)), ('test_tan', tan(x)), ('test_tanh', tanh(x))]
    numerical_tests = []
    for name, expr in name_expr:
        for xval in (0.2, 0.5, 0.8):
            expected = N(expr.subs(x, xval))
            numerical_tests.append((name, (xval,), expected, 1e-14))
    for lang, commands in valid_lang_commands:
        if lang.startswith('C'):
            name_expr_C = [('test_floor', floor(x)), ('test_ceil', ceiling(x))]
        else:
            name_expr_C = []
        run_test('intrinsic_math1', name_expr + name_expr_C, numerical_tests, lang, commands)
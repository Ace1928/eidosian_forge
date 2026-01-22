import tempfile
from sympy.core.numbers import pi, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions import assuming, Q
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.cfunctions import log2, exp2, expm1, log1p
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.codegen.rewriting import (
from sympy.testing.pytest import XFAIL, skip
from sympy.utilities import lambdify
from sympy.utilities._compilation import compile_link_import_strings, has_c
from sympy.utilities._compilation.util import may_xfail
@may_xfail
def test_compiled_ccode_with_rewriting():
    if not cython:
        skip('cython not installed.')
    if not has_c():
        skip('No C compiler found.')
    x = Symbol('x')
    about_two = 2 ** (58 / S(117)) * 3 ** (97 / S(117)) * 5 ** (4 / S(39)) * 7 ** (92 / S(117)) / S(30) * pi
    unchanged = 2 * exp(x) - about_two
    xval = S(10) ** (-11)
    ref = unchanged.subs(x, xval).n(19)
    rewritten = optimize(2 * exp(x) - about_two, [expm1_opt])
    NUMBER_OF_DIGITS = 25
    func_c = '\n#include <math.h>\n\ndouble func_unchanged(double x) {\n    return %(unchanged)s;\n}\ndouble func_rewritten(double x) {\n    return %(rewritten)s;\n}\n' % {'unchanged': ccode(unchanged.n(NUMBER_OF_DIGITS)), 'rewritten': ccode(rewritten.n(NUMBER_OF_DIGITS))}
    func_pyx = '\n#cython: language_level=3\ncdef extern double func_unchanged(double)\ncdef extern double func_rewritten(double)\ndef py_unchanged(x):\n    return func_unchanged(x)\ndef py_rewritten(x):\n    return func_rewritten(x)\n'
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([('func.c', func_c), ('_func.pyx', func_pyx)], build_dir=folder, compile_kwargs={'std': 'c99'})
        err_rewritten = abs(mod.py_rewritten(1e-11) - ref)
        err_unchanged = abs(mod.py_unchanged(1e-11) - ref)
        assert 1e-27 < err_rewritten < 1e-25
        assert 1e-19 < err_unchanged < 1e-16
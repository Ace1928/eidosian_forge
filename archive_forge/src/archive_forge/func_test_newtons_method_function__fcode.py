import tempfile
import sympy as sp
from sympy.codegen.ast import Assignment
from sympy.codegen.algorithms import newtons_method, newtons_method_function
from sympy.codegen.fnodes import bind_C
from sympy.codegen.futils import render_as_module as f_module
from sympy.codegen.pyutils import render_as_module as py_module
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.utilities._compilation import compile_link_import_strings, has_c, has_fortran
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip, raises
@may_xfail
def test_newtons_method_function__fcode():
    x = sp.Symbol('x', real=True)
    expr = sp.cos(x) - x ** 3
    func = newtons_method_function(expr, x, attrs=[bind_C(name='newton')])
    if not cython:
        skip('cython not installed.')
    if not has_fortran():
        skip('No Fortran compiler found.')
    f_mod = f_module([func], 'mod_newton')
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([('newton.f90', f_mod), ('_newton.pyx', '#cython: language_level={}\n'.format('3') + 'cdef extern double newton(double*)\ndef py_newton(double x):\n    return newton(&x)\n')], build_dir=folder)
        assert abs(mod.py_newton(0.5) - 0.865474033102) < 1e-12
import os
import tempfile
from sympy.core.symbol import (Symbol, symbols)
from sympy.codegen.ast import (
from sympy.codegen.fnodes import (
from sympy.codegen.futils import render_as_module
from sympy.core.expr import unchanged
from sympy.external import import_module
from sympy.printing.codeprinter import fcode
from sympy.utilities._compilation import has_fortran, compile_run_strings, compile_link_import_strings
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip, XFAIL
@may_xfail
def test_Module():
    x = Symbol('x', real=True)
    v_x = Variable.deduced(x)
    sq = FunctionDefinition(real, 'sqr', [v_x], [Return(x ** 2)])
    mod_sq = Module('mod_sq', [], [sq])
    sq_call = FunctionCall('sqr', [42.0])
    prg_sq = Program('foobar', [use('mod_sq', only=['sqr']), Print(['"Square of 42 = "', sq_call])])
    if not has_fortran():
        skip('No fortran compiler found.')
    (stdout, stderr), info = compile_run_strings([('mod_sq.f90', fcode(mod_sq, standard=90)), ('main.f90', fcode(prg_sq, standard=90))], clean=True)
    assert '42' in stdout
    assert str(42 ** 2) in stdout
    assert stderr == ''
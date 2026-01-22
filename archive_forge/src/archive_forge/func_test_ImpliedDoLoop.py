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
@XFAIL
@may_xfail
def test_ImpliedDoLoop():
    if not has_fortran():
        skip('No fortran compiler found.')
    a, i = symbols('a i', integer=True)
    idl = ImpliedDoLoop(i ** 3, i, -3, 3, 2)
    ac = ArrayConstructor([-28, idl, 28])
    a = array(a, dim=[':'], attrs=[allocatable])
    prog = Program('idlprog', [a.as_Declaration(), Assignment(a, ac), Print([a])])
    fsrc = fcode(prog, standard=2003, source_format='free')
    (stdout, stderr), info = compile_run_strings([('main.f90', fsrc)], clean=True)
    for numstr in '-28 -27 -1 1 27 28'.split():
        assert numstr in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK
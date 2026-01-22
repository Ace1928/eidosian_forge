import tempfile
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.utilities._compilation import compile_link_import_strings, has_c
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip
from sympy.codegen.ast import (
from sympy.codegen.cnodes import void, PreIncrement
from sympy.codegen.cutils import render_as_source_file
@may_xfail
def test_copying_function():
    if not np:
        skip('numpy not installed.')
    if not has_c():
        skip('No C compiler found.')
    if not cython:
        skip('Cython not found.')
    info = None
    with tempfile.TemporaryDirectory() as folder:
        mod, info = _render_compile_import(_mk_func1(), build_dir=folder)
        inp = np.arange(10.0)
        out = np.empty_like(inp)
        mod._our_test_function(inp, out)
        assert np.allclose(inp, out)
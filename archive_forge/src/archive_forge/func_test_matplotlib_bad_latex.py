from sympy.interactive.session import (init_ipython_session,
from sympy.core import Symbol, Rational, Integer
from sympy.external import import_module
from sympy.testing.pytest import raises
def test_matplotlib_bad_latex():
    app = init_ipython_session()
    app.run_cell('import IPython')
    app.run_cell('ip = get_ipython()')
    app.run_cell('inst = ip.instance()')
    app.run_cell('format = inst.display_formatter.format')
    app.run_cell('from sympy import init_printing, Matrix')
    app.run_cell("init_printing(use_latex='matplotlib')")
    app.run_cell("inst.display_formatter.formatters['image/png'].enabled = True")
    app.run_cell('import warnings')
    if int(ipython.__version__.split('.')[0]) < 2:
        app.run_cell("warnings.simplefilter('error')")
    else:
        app.run_cell("warnings.simplefilter('error', IPython.core.formatters.FormatterWarning)")
    app.run_cell('a = format(Matrix([1, 2, 3]))')
    app.run_cell('from sympy import Piecewise, Symbol, Eq')
    app.run_cell("x = Symbol('x'); pw = format(Piecewise((1, Eq(x, 0)), (0, True)))")
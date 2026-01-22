from sympy.interactive.session import (init_ipython_session,
from sympy.core import Symbol, Rational, Integer
from sympy.external import import_module
from sympy.testing.pytest import raises
def test_builtin_containers():
    app = init_ipython_session()
    app.run_cell('ip = get_ipython()')
    app.run_cell('inst = ip.instance()')
    app.run_cell('format = inst.display_formatter.format')
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell('from sympy import init_printing, Matrix')
    app.run_cell('init_printing(use_latex=True, use_unicode=False)')
    app.run_cell('a = format((True, False))')
    app.run_cell('import sys')
    app.run_cell('b = format(sys.flags)')
    app.run_cell('c = format((Matrix([1, 2]),))')
    if int(ipython.__version__.split('.')[0]) < 1:
        assert app.user_ns['a']['text/plain'] == '(True, False)'
        assert 'text/latex' not in app.user_ns['a']
        assert app.user_ns['b']['text/plain'][:10] == 'sys.flags('
        assert 'text/latex' not in app.user_ns['b']
        assert app.user_ns['c']['text/plain'] == ' [1]  \n([ ],)\n [2]  '
        assert app.user_ns['c']['text/latex'] == '$\\displaystyle \\left( \\left[\\begin{matrix}1\\\\2\\end{matrix}\\right],\\right)$'
    else:
        assert app.user_ns['a'][0]['text/plain'] == '(True, False)'
        assert 'text/latex' not in app.user_ns['a'][0]
        assert app.user_ns['b'][0]['text/plain'][:10] == 'sys.flags('
        assert 'text/latex' not in app.user_ns['b'][0]
        assert app.user_ns['c'][0]['text/plain'] == ' [1]  \n([ ],)\n [2]  '
        assert app.user_ns['c'][0]['text/latex'] == '$\\displaystyle \\left( \\left[\\begin{matrix}1\\\\2\\end{matrix}\\right],\\right)$'
from sympy.interactive.session import (init_ipython_session,
from sympy.core import Symbol, Rational, Integer
from sympy.external import import_module
from sympy.testing.pytest import raises
def test_print_builtin_option():
    app = init_ipython_session()
    app.run_cell('ip = get_ipython()')
    app.run_cell('inst = ip.instance()')
    app.run_cell('format = inst.display_formatter.format')
    app.run_cell('from sympy import Symbol')
    app.run_cell('from sympy import init_printing')
    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    if int(ipython.__version__.split('.')[0]) < 1:
        text = app.user_ns['a']['text/plain']
        raises(KeyError, lambda: app.user_ns['a']['text/latex'])
    else:
        text = app.user_ns['a'][0]['text/plain']
        raises(KeyError, lambda: app.user_ns['a'][0]['text/latex'])
    assert text in ('{pi: 3.14, n_i: 3}', '{nᵢ: 3, π: 3.14}', '{n_i: 3, pi: 3.14}', '{π: 3.14, nᵢ: 3}')
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell('init_printing(use_latex=True)')
    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    if int(ipython.__version__.split('.')[0]) < 1:
        text = app.user_ns['a']['text/plain']
        latex = app.user_ns['a']['text/latex']
    else:
        text = app.user_ns['a'][0]['text/plain']
        latex = app.user_ns['a'][0]['text/latex']
    assert text in ('{pi: 3.14, n_i: 3}', '{nᵢ: 3, π: 3.14}', '{n_i: 3, pi: 3.14}', '{π: 3.14, nᵢ: 3}')
    assert latex == '$\\displaystyle \\left\\{ n_{i} : 3, \\  \\pi : 3.14\\right\\}$'
    app.run_cell('    class WithOverload:\n        def _latex(self, printer):\n            return r"\\LaTeX"\n    ')
    app.run_cell('a = format((WithOverload(),))')
    if int(ipython.__version__.split('.')[0]) < 1:
        latex = app.user_ns['a']['text/latex']
    else:
        latex = app.user_ns['a'][0]['text/latex']
    assert latex == '$\\displaystyle \\left( \\LaTeX,\\right)$'
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell('init_printing(use_latex=True, print_builtin=False)')
    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    if int(ipython.__version__.split('.')[0]) < 1:
        text = app.user_ns['a']['text/plain']
        raises(KeyError, lambda: app.user_ns['a']['text/latex'])
    else:
        text = app.user_ns['a'][0]['text/plain']
        raises(KeyError, lambda: app.user_ns['a'][0]['text/latex'])
    assert text in ('{pi: 3.14, n_i: 3}', '{n_i: 3, pi: 3.14}')
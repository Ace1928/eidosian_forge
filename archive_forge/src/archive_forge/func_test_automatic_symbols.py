from sympy.interactive.session import (init_ipython_session,
from sympy.core import Symbol, Rational, Integer
from sympy.external import import_module
from sympy.testing.pytest import raises
def test_automatic_symbols():
    app = init_ipython_session()
    app.run_cell('from sympy import *')
    enable_automatic_symbols(app)
    symbol = 'verylongsymbolname'
    assert symbol not in app.user_ns
    app.run_cell('a = %s' % symbol, True)
    assert symbol not in app.user_ns
    app.run_cell('a = type(%s)' % symbol, True)
    assert app.user_ns['a'] == Symbol
    app.run_cell("%s = Symbol('%s')" % (symbol, symbol), True)
    assert symbol in app.user_ns
    app.run_cell('a = all == __builtin__.all', True)
    assert 'all' not in app.user_ns
    assert app.user_ns['a'] is True
    app.run_cell('import sympy')
    app.run_cell('a = factorial == sympy.factorial', True)
    assert app.user_ns['a'] is True
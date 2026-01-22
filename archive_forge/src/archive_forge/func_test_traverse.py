import pytest
from numpy.f2py.symbolic import (
from . import util
def test_traverse(self):
    x = as_symbol('x')
    y = as_symbol('y')
    z = as_symbol('z')
    f = as_symbol('f')

    def replace_visit(s, r=z):
        if s == x:
            return r
    assert x.traverse(replace_visit) == z
    assert y.traverse(replace_visit) == y
    assert z.traverse(replace_visit) == z
    assert f(y).traverse(replace_visit) == f(y)
    assert f(x).traverse(replace_visit) == f(z)
    assert f[y].traverse(replace_visit) == f[y]
    assert f[z].traverse(replace_visit) == f[z]
    assert (x + y + z).traverse(replace_visit) == 2 * z + y
    assert (x + f(y, x - z)).traverse(replace_visit) == z + f(y, as_number(0))
    assert as_eq(x, y).traverse(replace_visit) == as_eq(z, y)
    function_symbols = set()
    symbols = set()

    def collect_symbols(s):
        if s.op is Op.APPLY:
            oper = s.data[0]
            function_symbols.add(oper)
            if oper in symbols:
                symbols.remove(oper)
        elif s.op is Op.SYMBOL and s not in function_symbols:
            symbols.add(s)
    (x + f(y, x - z)).traverse(collect_symbols)
    assert function_symbols == {f}
    assert symbols == {x, y, z}

    def collect_symbols2(expr, symbols):
        if expr.op is Op.SYMBOL:
            symbols.add(expr)
    symbols = set()
    (x + f(y, x - z)).traverse(collect_symbols2, symbols)
    assert symbols == {x, y, z, f}

    def collect_symbols3(expr, symbols):
        if expr.op is Op.APPLY:
            return expr
        if expr.op is Op.SYMBOL:
            symbols.add(expr)
    symbols = set()
    (x + f(y, x - z)).traverse(collect_symbols3, symbols)
    assert symbols == {x}
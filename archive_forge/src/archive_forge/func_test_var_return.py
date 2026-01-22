from sympy.core.function import (Function, FunctionClass)
from sympy.core.symbol import (Symbol, var)
from sympy.testing.pytest import raises
def test_var_return():
    ns = {'var': var, 'raises': raises}
    "raises(ValueError, lambda: var(''))"
    v2 = eval("var('q')", ns)
    v3 = eval("var('q p')", ns)
    assert v2 == Symbol('q')
    assert v3 == (Symbol('q'), Symbol('p'))
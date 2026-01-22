from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.logic.boolalg import And
from sympy.core.symbol import Str
from sympy.unify.core import Compound, Variable
from sympy.unify.usympy import (deconstruct, construct, unify, is_associative,
from sympy.abc import x, y, z, n
def test_s_input():
    expr = Basic(S(1), S(2))
    a, b = map(Symbol, 'ab')
    pattern = Basic(a, b)
    assert list(unify(expr, pattern, {}, (a, b))) == [{a: 1, b: 2}]
    assert list(unify(expr, pattern, {a: 5}, (a, b))) == []
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
def test_unify_iter():
    expr = Add(1, 2, 3, evaluate=False)
    a, b, c = map(Symbol, 'abc')
    pattern = Add(a, c, evaluate=False)
    assert is_associative(deconstruct(pattern))
    assert is_commutative(deconstruct(pattern))
    result = list(unify(expr, pattern, {}, (a, c)))
    expected = [{a: 1, c: Add(2, 3, evaluate=False)}, {a: 1, c: Add(3, 2, evaluate=False)}, {a: 2, c: Add(1, 3, evaluate=False)}, {a: 2, c: Add(3, 1, evaluate=False)}, {a: 3, c: Add(1, 2, evaluate=False)}, {a: 3, c: Add(2, 1, evaluate=False)}, {a: Add(1, 2, evaluate=False), c: 3}, {a: Add(2, 1, evaluate=False), c: 3}, {a: Add(1, 3, evaluate=False), c: 2}, {a: Add(3, 1, evaluate=False), c: 2}, {a: Add(2, 3, evaluate=False), c: 1}, {a: Add(3, 2, evaluate=False), c: 1}]
    assert iterdicteq(result, expected)
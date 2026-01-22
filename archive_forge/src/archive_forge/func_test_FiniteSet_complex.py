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
def test_FiniteSet_complex():
    from sympy.sets.sets import FiniteSet
    a, b, c, x, y, z = symbols('a,b,c,x,y,z')
    expr = FiniteSet(Basic(S(1), x), y, Basic(x, z))
    pattern = FiniteSet(a, Basic(x, b))
    variables = (a, b)
    expected = ({b: 1, a: FiniteSet(y, Basic(x, z))}, {b: z, a: FiniteSet(y, Basic(S(1), x))})
    assert iterdicteq(unify(expr, pattern, variables=variables), expected)
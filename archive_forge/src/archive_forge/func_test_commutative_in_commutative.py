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
def test_commutative_in_commutative():
    from sympy.abc import a, b, c, d
    from sympy.functions.elementary.trigonometric import cos, sin
    eq = sin(3) * sin(4) * sin(5) + 4 * cos(3) * cos(4)
    pat = a * cos(b) * cos(c) + d * sin(b) * sin(c)
    assert next(unify(eq, pat, variables=(a, b, c, d)))
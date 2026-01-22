from sympy.core.logic import fuzzy_and
from sympy.core.sympify import _sympify
from sympy.multipledispatch import dispatch
from sympy.testing.pytest import XFAIL, raises
from sympy.assumptions.ask import Q
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (And, Implies, Not, Or, Xor)
from sympy.sets import Reals
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.core.relational import (Relational, Equality, Unequality,
from sympy.sets.sets import Interval, FiniteSet
from itertools import combinations
def test_relational_arithmetic():
    for cls in [Eq, Ne, Le, Lt, Ge, Gt]:
        rel = cls(x, y)
        raises(TypeError, lambda: 0 + rel)
        raises(TypeError, lambda: 1 * rel)
        raises(TypeError, lambda: 1 ** rel)
        raises(TypeError, lambda: rel ** 1)
        raises(TypeError, lambda: Add(0, rel))
        raises(TypeError, lambda: Mul(1, rel))
        raises(TypeError, lambda: Pow(1, rel))
        raises(TypeError, lambda: Pow(rel, 1))
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.ntheory.factor_ import factorint
from sympy.simplify.powsimp import powsimp
from sympy.core.function import _mexpand
from sympy.core.sorting import default_sort_key, ordered
from sympy.functions.elementary.trigonometric import sin
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import (diop_DN,
from sympy.testing.pytest import slow, raises, XFAIL
from sympy.utilities.iterables import (
def is_normal_transformation_ok(eq):
    A = transformation_to_normal(eq)
    X, Y, Z = A * Matrix([x, y, z])
    simplified = diop_simplify(eq.subs(zip((x, y, z), (X, Y, Z))))
    coeff = dict([reversed(t.as_independent(*[X, Y, Z])) for t in simplified.args])
    for term in [X * Y, Y * Z, X * Z]:
        if term in coeff.keys():
            return False
    return True
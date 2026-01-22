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
def is_pell_transformation_ok(eq):
    """
    Test whether X*Y, X, or Y terms are present in the equation
    after transforming the equation using the transformation returned
    by transformation_to_pell(). If they are not present we are good.
    Moreover, coefficient of X**2 should be a divisor of coefficient of
    Y**2 and the constant term.
    """
    A, B = transformation_to_DN(eq)
    u = (A * Matrix([X, Y]) + B)[0]
    v = (A * Matrix([X, Y]) + B)[1]
    simplified = diop_simplify(eq.subs(zip((x, y), (u, v))))
    coeff = dict([reversed(t.as_independent(*[X, Y])) for t in simplified.args])
    for term in [X * Y, X, Y]:
        if term in coeff.keys():
            return False
    for term in [X ** 2, Y ** 2, 1]:
        if term not in coeff.keys():
            coeff[term] = 0
    if coeff[X ** 2] != 0:
        return divisible(coeff[Y ** 2], coeff[X ** 2]) and divisible(coeff[1], coeff[X ** 2])
    return True
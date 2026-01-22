import itertools as it
from sympy.core.expr import unchanged
from sympy.core.function import Function
from sympy.core.numbers import I, oo, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.external import import_module
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.miscellaneous import (sqrt, cbrt, root, Min,
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.delta_functions import Heaviside
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, skip, ignore_warnings
def test_instantiation_evaluation():
    from sympy.abc import v, w, x, y, z
    assert Min(1, Max(2, x)) == 1
    assert Max(3, Min(2, x)) == 3
    assert Min(Max(x, y), Max(x, z)) == Max(x, Min(y, z))
    assert set(Min(Max(w, x), Max(y, z)).args) == {Max(w, x), Max(y, z)}
    assert Min(Max(x, y), Max(x, z), w) == Min(w, Max(x, Min(y, z)))
    A, B = (Min, Max)
    for i in range(2):
        assert A(x, B(x, y)) == x
        assert A(x, B(y, A(x, w, z))) == A(x, B(y, A(w, z)))
        A, B = (B, A)
    assert Min(w, Max(x, y), Max(v, x, z)) == Min(w, Max(x, Min(y, Max(v, z))))
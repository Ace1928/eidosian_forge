from sympy.core.numbers import (I, nan, oo, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, sign, transpose)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.simplify.simplify import signsimp
from sympy.testing.pytest import raises
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
def test_heaviside():
    assert Heaviside(-5) == 0
    assert Heaviside(1) == 1
    assert Heaviside(0) == S.Half
    assert Heaviside(0, x) == x
    assert unchanged(Heaviside, x, nan)
    assert Heaviside(0, nan) == nan
    h0 = Heaviside(x, 0)
    h12 = Heaviside(x, S.Half)
    h1 = Heaviside(x, 1)
    assert h0.args == h0.pargs == (x, 0)
    assert h1.args == h1.pargs == (x, 1)
    assert h12.args == (x, S.Half)
    assert h12.pargs == (x,)
    assert adjoint(Heaviside(x)) == Heaviside(x)
    assert adjoint(Heaviside(x - y)) == Heaviside(x - y)
    assert conjugate(Heaviside(x)) == Heaviside(x)
    assert conjugate(Heaviside(x - y)) == Heaviside(x - y)
    assert transpose(Heaviside(x)) == Heaviside(x)
    assert transpose(Heaviside(x - y)) == Heaviside(x - y)
    assert Heaviside(x).diff(x) == DiracDelta(x)
    assert Heaviside(x + I).is_Function is True
    assert Heaviside(I * x).is_Function is True
    raises(ArgumentIndexError, lambda: Heaviside(x).fdiff(2))
    raises(ValueError, lambda: Heaviside(I))
    raises(ValueError, lambda: Heaviside(2 + 3 * I))
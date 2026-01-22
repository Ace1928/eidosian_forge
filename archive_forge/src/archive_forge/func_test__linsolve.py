from sympy.testing.pytest import raises
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.abc import x, y, z
from sympy.polys.matrices.linsolve import _linsolve
from sympy.polys.solvers import PolyNonlinearError
def test__linsolve():
    assert _linsolve([], [x]) == {x: x}
    assert _linsolve([S.Zero], [x]) == {x: x}
    assert _linsolve([x - 1, x - 2], [x]) is None
    assert _linsolve([x - 1], [x]) == {x: 1}
    assert _linsolve([x - 1, y], [x, y]) == {x: 1, y: S.Zero}
    assert _linsolve([2 * I], [x]) is None
    raises(PolyNonlinearError, lambda: _linsolve([x * (1 + x)], [x]))
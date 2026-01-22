from sympy.core.function import expand
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import (cancel, factor)
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.radsimp import (collect, radsimp, rcollect)
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import (posify, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.abc import x, y, z
from sympy.testing.pytest import XFAIL
def test_posify():
    assert posify(A)[0].is_commutative is False
    for q in (A * B / A, (A * B / A) ** 2, (A * B) ** 2, A * B - B * A):
        p = posify(q)
        assert p[0].subs(p[1]) == q
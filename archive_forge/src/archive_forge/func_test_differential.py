from sympy.core import Lambda, Symbol, symbols
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r, R3_c, R3_s, R2_origin
from sympy.diffgeom import (Manifold, Patch, CoordSystem, Commutator, Differential, TensorProduct,
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin
from sympy.matrices import Matrix
from sympy.testing.pytest import raises, nocache_fail
from sympy.testing.pytest import warns_deprecated_sympy
def test_differential():
    xdy = R2.x * R2.dy
    dxdy = Differential(xdy)
    assert xdy.rcall(None) == xdy
    assert dxdy(R2.e_x, R2.e_y) == 1
    assert dxdy(R2.e_x, R2.x * R2.e_y) == R2.x
    assert Differential(dxdy) == 0
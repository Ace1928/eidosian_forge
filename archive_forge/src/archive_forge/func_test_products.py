from sympy.core import Lambda, Symbol, symbols
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r, R3_c, R3_s, R2_origin
from sympy.diffgeom import (Manifold, Patch, CoordSystem, Commutator, Differential, TensorProduct,
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin
from sympy.matrices import Matrix
from sympy.testing.pytest import raises, nocache_fail
from sympy.testing.pytest import warns_deprecated_sympy
def test_products():
    assert TensorProduct(R2.dx, R2.dy)(R2.e_x, R2.e_y) == R2.dx(R2.e_x) * R2.dy(R2.e_y) == 1
    assert TensorProduct(R2.dx, R2.dy)(None, R2.e_y) == R2.dx
    assert TensorProduct(R2.dx, R2.dy)(R2.e_x, None) == R2.dy
    assert TensorProduct(R2.dx, R2.dy)(R2.e_x) == R2.dy
    assert TensorProduct(R2.x, R2.dx) == R2.x * R2.dx
    assert TensorProduct(R2.e_x, R2.e_y)(R2.x, R2.y) == R2.e_x(R2.x) * R2.e_y(R2.y) == 1
    assert TensorProduct(R2.e_x, R2.e_y)(None, R2.y) == R2.e_x
    assert TensorProduct(R2.e_x, R2.e_y)(R2.x, None) == R2.e_y
    assert TensorProduct(R2.e_x, R2.e_y)(R2.x) == R2.e_y
    assert TensorProduct(R2.x, R2.e_x) == R2.x * R2.e_x
    assert TensorProduct(R2.dx, R2.e_y)(R2.e_x, R2.y) == R2.dx(R2.e_x) * R2.e_y(R2.y) == 1
    assert TensorProduct(R2.dx, R2.e_y)(None, R2.y) == R2.dx
    assert TensorProduct(R2.dx, R2.e_y)(R2.e_x, None) == R2.e_y
    assert TensorProduct(R2.dx, R2.e_y)(R2.e_x) == R2.e_y
    assert TensorProduct(R2.x, R2.e_x) == R2.x * R2.e_x
    assert TensorProduct(R2.e_x, R2.dy)(R2.x, R2.e_y) == R2.e_x(R2.x) * R2.dy(R2.e_y) == 1
    assert TensorProduct(R2.e_x, R2.dy)(None, R2.e_y) == R2.e_x
    assert TensorProduct(R2.e_x, R2.dy)(R2.x, None) == R2.dy
    assert TensorProduct(R2.e_x, R2.dy)(R2.x) == R2.dy
    assert TensorProduct(R2.e_y, R2.e_x)(R2.x ** 2 + R2.y ** 2, R2.x ** 2 + R2.y ** 2) == 4 * R2.x * R2.y
    assert WedgeProduct(R2.dx, R2.dy)(R2.e_x, R2.e_y) == 1
    assert WedgeProduct(R2.e_x, R2.e_y)(R2.x, R2.y) == 1
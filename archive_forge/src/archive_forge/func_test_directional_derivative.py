from sympy.core.function import Derivative
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.simplify import simplify
from sympy.core.symbol import symbols
from sympy.core import S
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.vector.vector import Dot
from sympy.vector.operators import curl, divergence, gradient, Gradient, Divergence, Cross
from sympy.vector.deloperator import Del
from sympy.vector.functions import (is_conservative, is_solenoidal,
from sympy.testing.pytest import raises
def test_directional_derivative():
    assert directional_derivative(C.x * C.y * C.z, 3 * C.i + 4 * C.j + C.k) == C.x * C.y + 4 * C.x * C.z + 3 * C.y * C.z
    assert directional_derivative(5 * C.x ** 2 * C.z, 3 * C.i + 4 * C.j + C.k) == 5 * C.x ** 2 + 30 * C.x * C.z
    assert directional_derivative(5 * C.x ** 2 * C.z, 4 * C.j) is S.Zero
    D = CoordSys3D('D', 'spherical', variable_names=['r', 'theta', 'phi'], vector_names=['e_r', 'e_theta', 'e_phi'])
    r, theta, phi = D.base_scalars()
    e_r, e_theta, e_phi = D.base_vectors()
    assert directional_derivative(r ** 2 * e_r, e_r) == 2 * r * e_r
    assert directional_derivative(5 * r ** 2 * phi, 3 * e_r + 4 * e_theta + e_phi) == 5 * r ** 2 + 30 * r * phi
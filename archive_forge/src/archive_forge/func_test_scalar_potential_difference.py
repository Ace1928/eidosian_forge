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
def test_scalar_potential_difference():
    point1 = C.origin.locate_new('P1', 1 * i + 2 * j + 3 * k)
    point2 = C.origin.locate_new('P2', 4 * i + 5 * j + 6 * k)
    genericpointC = C.origin.locate_new('RP', x * i + y * j + z * k)
    genericpointP = P.origin.locate_new('PP', P.x * P.i + P.y * P.j + P.z * P.k)
    assert scalar_potential_difference(S.Zero, C, point1, point2) == 0
    assert scalar_potential_difference(scalar_field, C, C.origin, genericpointC) == scalar_field
    assert scalar_potential_difference(grad_field, C, C.origin, genericpointC) == scalar_field
    assert scalar_potential_difference(grad_field, C, point1, point2) == 948
    assert scalar_potential_difference(y * z * i + x * z * j + x * y * k, C, point1, genericpointC) == x * y * z - 6
    potential_diff_P = 2 * P.z * (P.x * sin(q) + P.y * cos(q)) * (P.x * cos(q) - P.y * sin(q)) ** 2
    assert scalar_potential_difference(grad_field, P, P.origin, genericpointP).simplify() == potential_diff_P.simplify()
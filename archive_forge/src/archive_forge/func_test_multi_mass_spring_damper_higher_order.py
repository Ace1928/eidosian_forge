import sympy.physics.mechanics.models as models
from sympy.core.backend import (cos, sin, Matrix, symbols, zeros)
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols)
def test_multi_mass_spring_damper_higher_order():
    c0, k0, m0 = symbols('c0 k0 m0')
    c1, k1, m1 = symbols('c1 k1 m1')
    c2, k2, m2 = symbols('c2 k2 m2')
    v0, x0 = dynamicsymbols('v0 x0')
    v1, x1 = dynamicsymbols('v1 x1')
    v2, x2 = dynamicsymbols('v2 x2')
    kane1 = models.multi_mass_spring_damper(3)
    massmatrix1 = Matrix([[m0 + m1 + m2, m1 + m2, m2], [m1 + m2, m1 + m2, m2], [m2, m2, m2]])
    forcing1 = Matrix([[-c0 * v0 - k0 * x0], [-c1 * v1 - k1 * x1], [-c2 * v2 - k2 * x2]])
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(3)
    assert simplify(forcing1 - kane1.forcing) == Matrix([0, 0, 0])
from sympy.core.symbol import symbols
from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy.core.backend import expand, zeros, _simplify_matrix
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_deprecated_set_potential_energy():
    m, g, h = symbols('m g h')
    A = ReferenceFrame('A')
    P = Point('P')
    I = Dyadic(0)
    B = RigidBody('B', P, A, m, (I, P))
    with warns_deprecated_sympy():
        B.set_potential_energy(m * g * h)
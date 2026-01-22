from sympy.core.backend import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
from sympy.physics.mechanics.functions import (gravity, center_of_mass,
from sympy.testing.pytest import raises
def test_Lagrangian():
    M, m, g, h = symbols('M m g h')
    N = ReferenceFrame('N')
    O = Point('O')
    O.set_vel(N, 0 * N.x)
    P = O.locatenew('P', 1 * N.x)
    P.set_vel(N, 10 * N.x)
    Pa = Particle('Pa', P, 1)
    Ac = O.locatenew('Ac', 2 * N.y)
    Ac.set_vel(N, 5 * N.y)
    a = ReferenceFrame('a')
    a.set_ang_vel(N, 10 * N.z)
    I = outer(N.z, N.z)
    A = RigidBody('A', Ac, a, 20, (I, Ac))
    Pa.potential_energy = m * g * h
    A.potential_energy = M * g * h
    raises(TypeError, lambda: Lagrangian(A, A, Pa))
    raises(TypeError, lambda: Lagrangian(N, N, Pa))
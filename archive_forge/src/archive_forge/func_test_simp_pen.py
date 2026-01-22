from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises
def test_simp_pen():
    q, u = dynamicsymbols('q u')
    qd, ud = dynamicsymbols('q u ', 1)
    l, m, g = symbols('l m g')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q, N.z])
    A.set_ang_vel(N, qd * N.z)
    O = Point('O')
    O.set_vel(N, 0)
    P = O.locatenew('P', l * A.x)
    P.v2pt_theory(O, N, A)
    Pa = Particle('Pa', P, m)
    Pa.potential_energy = -m * g * l * cos(q)
    L = Lagrangian(N, Pa)
    lm = LagrangesMethod(L, [q])
    lm.form_lagranges_equations()
    RHS = lm.rhs()
    assert RHS[1] == -g * sin(q) / l
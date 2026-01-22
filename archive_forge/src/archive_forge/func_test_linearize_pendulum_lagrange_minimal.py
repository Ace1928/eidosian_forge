from sympy.core.backend import (symbols, Matrix, cos, sin, atan, sqrt,
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point,\
from sympy.testing.pytest import slow
def test_linearize_pendulum_lagrange_minimal():
    q1 = dynamicsymbols('q1')
    q1d = dynamicsymbols('q1', 1)
    L, m, t = symbols('L, m, t')
    g = 9.8
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)
    A = N.orientnew('A', 'axis', [q1, N.z])
    A.set_ang_vel(N, q1d * N.z)
    P = pN.locatenew('P', L * A.x)
    P.v2pt_theory(pN, N, A)
    pP = Particle('pP', P, m)
    Lag = Lagrangian(N, pP)
    LM = LagrangesMethod(Lag, [q1], forcelist=[(P, m * g * N.x)], frame=N)
    LM.form_lagranges_equations()
    A, B, inp_vec = LM.linearize([q1], [q1d], A_and_B=True)
    assert _simplify_matrix(A) == Matrix([[0, 1], [-9.8 * cos(q1) / L, 0]])
    assert B == Matrix([])
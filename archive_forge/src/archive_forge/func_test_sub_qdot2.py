from sympy.core.backend import cos, Matrix, sin, zeros, tan, pi, symbols
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.solvers.solvers import solve
from sympy.physics.mechanics import (cross, dot, dynamicsymbols,
def test_sub_qdot2():
    g, m, Px, Py, Pz, R, t = symbols('g m Px Py Pz R t')
    q = dynamicsymbols('q:5')
    qd = dynamicsymbols('q:5', level=1)
    u = dynamicsymbols('u:5')
    A = ReferenceFrame('A')
    B_prime = A.orientnew('B_prime', 'Axis', [q[0], A.z])
    B = B_prime.orientnew('B', 'Axis', [pi / 2 - q[1], B_prime.x])
    C = B.orientnew('C', 'Axis', [q[2], B.z])
    pO = Point('O')
    pO.set_vel(A, 0)
    pR = pO.locatenew('R', q[3] * A.x + q[4] * A.y)
    pR.set_vel(A, pR.pos_from(pO).diff(t, A))
    pR.set_vel(B, 0)
    pC_hat = pR.locatenew('C^', 0)
    pC_hat.set_vel(C, 0)
    pCs = pC_hat.locatenew('C*', R * B.y)
    pCs.set_vel(C, 0)
    pCs.set_vel(B, 0)
    pCs.v2pt_theory(pR, A, B)
    pC_hat.v2pt_theory(pCs, A, C)
    R_C_hat = Px * A.x + Py * A.y + Pz * A.z
    R_Cs = -m * g * A.z
    forces = [(pC_hat, R_C_hat), (pCs, R_Cs)]
    u_expr = [C.ang_vel_in(A) & uv for uv in B]
    u_expr += qd[3:]
    kde = [ui - e for ui, e in zip(u, u_expr)]
    km1 = KanesMethod(A, q, u, kde)
    fr1, _ = km1.kanes_equations([], forces)
    u_indep = u[:3]
    u_dep = list(set(u) - set(u_indep))
    vc = [pC_hat.vel(A) & uv for uv in [A.x, A.y]]
    km2 = KanesMethod(A, q, u_indep, kde, u_dependent=u_dep, velocity_constraints=vc)
    fr2, _ = km2.kanes_equations([], forces)
    fr1_expected = Matrix([-R * g * m * sin(q[1]), -R * (Px * cos(q[0]) + Py * sin(q[0])) * tan(q[1]), R * (Px * cos(q[0]) + Py * sin(q[0])), Px, Py])
    fr2_expected = Matrix([-R * g * m * sin(q[1]), 0, 0])
    assert trigsimp(fr1.expand()) == trigsimp(fr1_expected.expand())
    assert trigsimp(fr2.expand()) == trigsimp(fr2_expected.expand())
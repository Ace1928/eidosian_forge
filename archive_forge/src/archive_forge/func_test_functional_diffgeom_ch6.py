from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r
from sympy.diffgeom import intcurve_series, Differential, WedgeProduct
from sympy.core import symbols, Function, Derivative
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin, cos
from sympy.matrices import Matrix
def test_functional_diffgeom_ch6():
    u0, u1, u2, v0, v1, v2, w0, w1, w2 = symbols('u0:3, v0:3, w0:3', real=True)
    u = u0 * R2.e_x + u1 * R2.e_y
    v = v0 * R2.e_x + v1 * R2.e_y
    wp = WedgeProduct(R2.dx, R2.dy)
    assert wp(u, v) == u0 * v1 - u1 * v0
    u = u0 * R3_r.e_x + u1 * R3_r.e_y + u2 * R3_r.e_z
    v = v0 * R3_r.e_x + v1 * R3_r.e_y + v2 * R3_r.e_z
    w = w0 * R3_r.e_x + w1 * R3_r.e_y + w2 * R3_r.e_z
    wp = WedgeProduct(R3_r.dx, R3_r.dy, R3_r.dz)
    assert wp(u, v, w) == Matrix(3, 3, [u0, u1, u2, v0, v1, v2, w0, w1, w2]).det()
    a, b, c = symbols('a, b, c', cls=Function)
    a_f = a(R3_r.x, R3_r.y, R3_r.z)
    b_f = b(R3_r.x, R3_r.y, R3_r.z)
    c_f = c(R3_r.x, R3_r.y, R3_r.z)
    theta = a_f * R3_r.dx + b_f * R3_r.dy + c_f * R3_r.dz
    dtheta = Differential(theta)
    da = Differential(a_f)
    db = Differential(b_f)
    dc = Differential(c_f)
    expr = dtheta - WedgeProduct(da, R3_r.dx) - WedgeProduct(db, R3_r.dy) - WedgeProduct(dc, R3_r.dz)
    assert expr.rcall(R3_r.e_x, R3_r.e_y) == 0
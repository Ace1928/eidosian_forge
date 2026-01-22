from sympy.core.function import expand
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.mechanics import (PinJoint, JointsMethod, Body, KanesMethod,
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.testing.pytest import raises
from sympy.core.backend import zeros
from sympy.utilities.lambdify import lambdify
from sympy.solvers.solvers import solve
def test_simple_pedulum():
    l, m, g = symbols('l m g')
    C = Body('C')
    b = Body('b', mass=m)
    q = dynamicsymbols('q')
    P = PinJoint('P', C, b, speeds=q.diff(t), coordinates=q, child_point=-l * b.x, joint_axis=C.z)
    b.potential_energy = -m * g * l * cos(q)
    method = JointsMethod(C, P)
    method.form_eoms(LagrangesMethod)
    rhs = method.rhs()
    assert rhs[1] == -g * sin(q) / l
from sympy.core.backend import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
from sympy.physics.mechanics.functions import (gravity, center_of_mass,
from sympy.testing.pytest import raises
def test_find_dynamicsymbols():
    a, b = symbols('a, b')
    x, y, z = dynamicsymbols('x, y, z')
    expr = Matrix([[a * x + b, x * y.diff() + y], [x.diff().diff(), z + sin(z.diff())]])
    sol = {x, y.diff(), y, x.diff().diff(), z, z.diff()}
    assert find_dynamicsymbols(expr) == sol
    exclude_list = [x, y, z]
    sol = {y.diff(), x.diff().diff(), z.diff()}
    assert find_dynamicsymbols(expr, exclude=exclude_list) == sol
    d, e, f = dynamicsymbols('d, e, f')
    A = ReferenceFrame('A')
    v = d * A.x + e * A.y + f * A.z
    sol = {d, e, f}
    assert find_dynamicsymbols(v, reference_frame=A) == sol
    raises(ValueError, lambda: find_dynamicsymbols(v))
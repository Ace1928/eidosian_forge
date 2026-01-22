from sympy.core.symbol import symbols
from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy.core.backend import expand, zeros, _simplify_matrix
from sympy.testing.pytest import raises, warns_deprecated_sympy
Consider a pendulum of length OA = 2a, of mass m as a rigid body of
    center of mass G (OG = a) which turn around (O,z). The angle between the
    reference frame R and the rod is q.  The inertia of the body is I =
    (G,0,ma^2/3,ma^2/3). 
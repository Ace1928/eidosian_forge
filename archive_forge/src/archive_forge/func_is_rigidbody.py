from sympy.core.backend import Symbol
from sympy.physics.vector import Point, Vector, ReferenceFrame, Dyadic
from sympy.physics.mechanics import RigidBody, Particle, inertia
@property
def is_rigidbody(self):
    if hasattr(self, '_inertia'):
        return True
    return False
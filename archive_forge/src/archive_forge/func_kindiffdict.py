from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
def kindiffdict(self):
    """Returns a dictionary mapping q' to u."""
    if not self._qdot_u_map:
        raise AttributeError('Create an instance of KanesMethod with kinematic differential equations to use this method.')
    return self._qdot_u_map
from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
@property
def forcing(self):
    """The forcing vector of the system."""
    if not self._fr or not self._frstar:
        raise ValueError('Need to compute Fr, Fr* first.')
    return -Matrix([self._f_d, self._f_dnh])
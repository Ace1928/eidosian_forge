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
def mass_matrix_kin(self):
    """The kinematic "mass matrix" $\\mathbf{k_{k\\dot{q}}}$ of the system."""
    return self._k_kqdot if self.explicit_kinematics else self._k_kqdot_implicit
from sympy.core.backend import sympify
from sympy.physics.vector import Point, ReferenceFrame, Dyadic
from sympy.utilities.exceptions import sympy_deprecation_warning
@central_inertia.setter
def central_inertia(self, I):
    if not isinstance(I, Dyadic):
        raise TypeError('RigidBody inertia must be a Dyadic object.')
    self.inertia = (I, self.masscenter)
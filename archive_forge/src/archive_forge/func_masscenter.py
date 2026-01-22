from sympy.core.backend import sympify
from sympy.physics.vector import Point, ReferenceFrame, Dyadic
from sympy.utilities.exceptions import sympy_deprecation_warning
@masscenter.setter
def masscenter(self, p):
    if not isinstance(p, Point):
        raise TypeError('RigidBody center of mass must be a Point object.')
    self._masscenter = p
from sympy.core.backend import sympify
from sympy.physics.vector import Point, ReferenceFrame, Dyadic
from sympy.utilities.exceptions import sympy_deprecation_warning
def kinetic_energy(self, frame):
    """Kinetic energy of the rigid body.

        Explanation
        ===========

        The kinetic energy, T, of a rigid body, B, is given by:

        ``T = 1/2 * (dot(dot(I, w), w) + dot(m * v, v))``

        where I and m are the central inertia dyadic and mass of rigid body B,
        respectively, omega is the body's angular velocity and v is the
        velocity of the body's mass center in the supplied ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The RigidBody's angular velocity and the velocity of it's mass
            center are typically defined with respect to an inertial frame but
            any relevant frame in which the velocities are known can be supplied.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody
        >>> from sympy import symbols
        >>> M, v, r, omega = symbols('M v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> I = outer (b.x, b.x)
        >>> inertia_tuple = (I, P)
        >>> B = RigidBody('B', P, b, M, inertia_tuple)
        >>> B.kinetic_energy(N)
        M*v**2/2 + omega**2/2

        """
    rotational_KE = self.frame.ang_vel_in(frame) & (self.central_inertia & self.frame.ang_vel_in(frame)) / sympify(2)
    translational_KE = self.mass * (self.masscenter.vel(frame) & self.masscenter.vel(frame)) / sympify(2)
    return rotational_KE + translational_KE
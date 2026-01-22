from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
def set_acc(self, frame, value):
    """Used to set the acceleration of this Point in a ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which this point's acceleration is defined
        value : Vector
            The vector value of this point's acceleration in the frame

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_acc(N, 10 * N.x)
        >>> p1.acc(N)
        10*N.x

        """
    if value == 0:
        value = Vector(0)
    value = _check_vector(value)
    _check_frame(frame)
    self._acc_dict.update({frame: value})
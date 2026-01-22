from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import _path
from sympy.core.cache import cacheit
def express_coordinates(self, coordinate_system):
    """
        Returns the Cartesian/rectangular coordinates of this point
        wrt the origin of the given CoordSys3D instance.

        Parameters
        ==========

        coordinate_system : CoordSys3D
            The coordinate system to express the coordinates of this
            Point in.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> p2 = p1.locate_new('p2', 5 * N.j)
        >>> p2.express_coordinates(N)
        (10, 5, 0)

        """
    pos_vect = self.position_wrt(coordinate_system.origin)
    return tuple(pos_vect.to_matrix(coordinate_system))
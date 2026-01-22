from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def lift_vertex_positions_of_horotriangles(self):
    """
        After developing an incomplete cusp with
        add_vertex_positions_to_horotriangles, this function moves the
        vertex positions first to zero the fixed point (see
        move_ffixed_point_to_zero) and computes logarithms for all the
        vertex positions of the horotriangles in the Euclidean plane
        in a consistent manner. These logarithms are written to a
        dictionary lifted_vertex_positions on the HoroTriangle's.

        For an incomplete cusp, the respective value in lifted_vertex_positions
        will be None.

        The three logarithms of the vertex positions of a triangle are only
        defined up to adding mu Z + lambda Z where mu and lambda are the
        logarithmic holonomies of the meridian and longitude.
        """
    self.move_fixed_point_to_zero()
    for cusp in self.mcomplex.Vertices:
        self._lift_one_cusp_vertex_positions(cusp)
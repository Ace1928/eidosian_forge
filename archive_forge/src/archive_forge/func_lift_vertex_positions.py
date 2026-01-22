from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def lift_vertex_positions(self, lifted_position):
    """
        Lift the vertex positions of this triangle. lifted_position is
        used as a guide what branch of the logarithm to use.

        The lifted position is computed as the log of the vertex
        position where it is assumed that the fixed point of the
        holonomy is at the origin.  The branch of the logarithm
        closest to lifted_position is used.
        """
    NumericalField = lifted_position.parent()
    twoPi = 2 * NumericalField.pi()
    I = NumericalField(1j)

    def adjust_log(z):
        logZ = log(z)
        return logZ + ((lifted_position - logZ) / twoPi).imag().round() * twoPi * I
    self.lifted_vertex_positions = {edge: adjust_log(position) for edge, position in self.vertex_positions.items()}
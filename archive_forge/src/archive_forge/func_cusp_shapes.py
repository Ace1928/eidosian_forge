from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def cusp_shapes(self):
    """
        Compute the cusp shapes as conjugate of the quotient of the translations
        corresponding to the longitude and meridian for each cusp (SnapPea
        kernel convention).
        """
    self.compute_translations()
    return [ComplexCuspCrossSection._compute_cusp_shape(vertex) for vertex in self.mcomplex.Vertices]
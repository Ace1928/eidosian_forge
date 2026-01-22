from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def scale_cusps(self, scales):
    """
        Scale each cusp by Euclidean dilation by values in given array.
        """
    for cusp, scale in zip(self.mcomplex.Vertices, scales):
        CuspCrossSectionBase._scale_cusp(cusp, scale)
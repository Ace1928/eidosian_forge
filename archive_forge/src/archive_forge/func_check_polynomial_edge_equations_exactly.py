from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def check_polynomial_edge_equations_exactly(self):
    """
        Check that the polynomial edge equations are fulfilled exactly.

        We use the conjugate inverse to support non-orientable manifolds.
        """
    for edge in self.mcomplex.Edges:
        val = 1
        for tet, perm in edge.embeddings():
            val *= CuspCrossSectionBase._shape_for_edge_embedding(tet, perm)
        if not val == 1:
            raise EdgeEquationExactVerifyError(val)
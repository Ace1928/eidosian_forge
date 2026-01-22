from ...sage_helper import _within_sage
from ...math_basics import correct_max
from ...snap.kernel_structures import *
from ...snap.fundamental_polyhedron import *
from ...snap.mcomplex_base import *
from ...snap.t3mlite import simplex
from ...snap import t3mlite as t3m
from ...exceptions import InsufficientPrecisionError
from ..cuspCrossSection import ComplexCuspCrossSection
from ..upper_halfspace.ideal_point import *
from ..interval_tree import *
from .cusp_translate_engine import *
import heapq
def get_initial_cusp_triangle(self):
    tet = self.mcomplex.Tetrahedra[0]
    CIF = tet.ShapeParameters[simplex.E01].parent()
    m = matrix.identity(CIF, 2)
    corner = self.vertex_at_infinity.Corners[0]
    return (corner.Tetrahedron.Index, corner.Subsimplex, m)
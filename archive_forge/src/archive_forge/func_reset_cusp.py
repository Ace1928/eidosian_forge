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
def reset_cusp(self, cusp_index):
    self.intervalTree = IntervalTree()
    self.unglued_generator_heapq = []
    original_vertex = self.original_mcomplex.Vertices[cusp_index]
    original_corner = original_vertex.Corners[0]
    tet = self.mcomplex.Tetrahedra[original_corner.Tetrahedron.Index]
    RIF = tet.ShapeParameters[simplex.E01].real().parent()
    self.max_horosphere_height_for_cusp = self.num_cusps * [RIF(0)]
    self.vertex_at_infinity = tet.Class[original_corner.Subsimplex]
    f = FundamentalPolyhedronEngine(self.mcomplex)
    init_vertices = CuspTilingEngine.get_init_vertices(self.vertex_at_infinity)
    f.visit_tetrahedra_to_compute_vertices(tet, init_vertices)
    f.compute_matrices(normalize_matrices=False)
    self.baseTetInRadius, self.baseTetInCenter = compute_inradius_and_incenter([tet.Class[v].IdealPoint for v in simplex.ZeroSubsimplices])
    translations = self.translations[cusp_index]
    self.cuspTranslateEngine = CuspTranslateEngine(*translations)
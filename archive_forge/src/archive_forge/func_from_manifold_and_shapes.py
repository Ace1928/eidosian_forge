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
@staticmethod
def from_manifold_and_shapes(snappyManifold, shapes):
    c = ComplexCuspCrossSection.fromManifoldAndShapes(snappyManifold, shapes)
    c.ensure_std_form(allow_scaling_up=True)
    c.compute_translations()
    m = c.mcomplex
    cusp_areas = c.cusp_areas()
    translations = [vertex.Translations for vertex in m.Vertices]
    t = TransferKernelStructuresEngine(m, snappyManifold)
    t.choose_and_transfer_generators(compute_corners=True, centroid_at_origin=False)
    f = FundamentalPolyhedronEngine(m)
    f.unglue()
    original_mcomplex = t3m.Mcomplex(snappyManifold)
    t = TransferKernelStructuresEngine(original_mcomplex, snappyManifold)
    t.reindex_cusps_and_transfer_peripheral_curves()
    t.choose_and_transfer_generators(compute_corners=False, centroid_at_origin=False)
    return CuspTilingEngine(m, original_mcomplex, cusp_areas, translations)
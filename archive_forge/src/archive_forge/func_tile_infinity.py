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
def tile_infinity(self):
    pending_cusp_triangles = [self.get_initial_cusp_triangle()]
    processed_cusp_triangles = set()
    unprocessed_vertices = []
    self.horosphere_at_inf_height = None
    while pending_cusp_triangles:
        cusp_triangle = pending_cusp_triangles.pop()
        tet_index, V, m = cusp_triangle
        key = (tet_index, V)
        if key not in processed_cusp_triangles:
            processed_cusp_triangles.add(key)
            tet = self.mcomplex.Tetrahedra[tet_index]
            tile = self.find_tile(m)
            if not tile:
                tile = self.create_tile(m)
                unprocessed_vertices.append((tile, self.unglued_vertices_for_tile(tile)))
            vertex = tet.Class[V]
            tile.vertices_at_infinity.add(vertex)
            if self.horosphere_at_inf_height is None:
                self.horosphere_at_inf_height = tile.height_of_horosphere(vertex, is_at_infinity=True)
            for neighboring_triangle in self.get_neighboring_cusp_triangles(cusp_triangle):
                pending_cusp_triangles.append(neighboring_triangle)
    for tile, pending_vertices in unprocessed_vertices:
        for vertex in pending_vertices:
            if vertex not in tile.vertices_at_infinity:
                self.account_horosphere_height(tile, vertex)
        for g in self.unglued_generators_for_tile(tile):
            self.record_unglued_generator(tile, g)
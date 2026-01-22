from ... import sage_helper
from .. import t3mlite as t3m
from . import link, dual_cellulation
def peripheral_curve_from_snappy(dual_cell, snappy_data):
    D = dual_cell
    T = D.dual_triangulation
    M = T.parent_triangulation
    data = snappy_data
    weights = len(D.edges) * [0]
    for tet_index, tet in enumerate(M.Tetrahedra):
        for vert_index, V in enumerate(t3m.ZeroSubsimplices):
            triangle = tet.CuspCorners[V]
            sides = triangle.oriented_sides()
            for tri_edge_index, tet_edge in enumerate(link.TruncatedSimplexCorners[V]):
                tet_face_index = t3m.ZeroSubsimplices.index(tet_edge ^ V)
                side = sides[tri_edge_index]
                global_edge = side.edge()
                if global_edge.orientation_with_respect_to(side) > 0:
                    dual_edge = D.from_original[global_edge]
                    weight = data[tet_index][4 * vert_index + tet_face_index]
                    weights[dual_edge.index] = -weight
    total_raw_weights = sum([sum((abs(x) for x in row)) for row in data])
    assert 2 * sum((abs(w) for w in weights)) == total_raw_weights
    return dual_cellulation.OneCycle(D, weights)
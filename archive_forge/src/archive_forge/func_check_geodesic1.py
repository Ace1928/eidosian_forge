from ..snap.t3mlite import simplex
from ..hyperboloid import *
def check_geodesic1(tets):
    RF = tets[0].O13_matrices[simplex.F0].base_ring()
    for tet in tets:
        for geodesic_segment in tet.geodesic_pieces:
            if geodesic_segment.tet is not tet:
                raise Exception('Geodesic tet inconsistency')
            for ptInClass in geodesic_segment.endpoints:
                if ptInClass.subsimplex in simplex.ZeroSubsimplices:
                    check_points_equal(ptInClass.r13_point, tet.R13_vertices[ptInClass.subsimplex])
                elif abs(r13_dot(ptInClass.r13_point, tet.R13_planes[ptInClass.subsimplex])) > RF(1e-10):
                    raise Exception('Point not on plane')
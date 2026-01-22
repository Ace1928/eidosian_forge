from .geodesic_info import GeodesicInfo
from .geometric_structure import Filling, FillingMatrix
from ..snap.t3mlite import Mcomplex, simplex
from typing import Tuple, Optional, Sequence
def index_geodesics_and_add_post_drill_infos(geodesics: Sequence[GeodesicInfo], mcomplex: Mcomplex) -> None:
    all_reindexed_verts = set((g.core_curve_cusp for g in geodesics if g.core_curve_cusp))
    old_vertices = [v for v in mcomplex.Vertices if v not in all_reindexed_verts]
    for i, v in enumerate(old_vertices):
        v.post_drill_info = CuspPostDrillInfo(index=i, filling=v.filling_matrix[0])
    n = len(old_vertices)
    for i, g in enumerate(geodesics):
        if g.core_curve_cusp:
            g.core_curve_cusp.post_drill_info = CuspPostDrillInfo(index=n + i, peripheral_matrix=_multiply_filling_matrix(g.core_curve_cusp.filling_matrix, g.core_curve_direction))
        else:
            g.index = n + i
    for tet in mcomplex.Tetrahedra:
        tet.post_drill_infos = {V: tet.Class[V].post_drill_info for V in simplex.ZeroSubsimplices}
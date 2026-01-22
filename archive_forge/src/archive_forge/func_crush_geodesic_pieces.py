from .cusps import CuspPostDrillInfo
from .tracing import GeodesicPiece
from .peripheral_curves import install_peripheral_curves
from ..snap.t3mlite import Tetrahedron, Perm4, Mcomplex, simplex
from typing import Dict, Tuple, List, Sequence
def crush_geodesic_pieces(tetrahedra: Sequence[Tetrahedron]) -> Mcomplex:
    """
    Given tetrahedra produced by traverse_geodesics_to_subdivide,
    compute the barycentric subdivision and crush all tetrahedra in the
    barycentric subdivision that are adjacent to an edge that coincides with a
    GeodesicPiece.

    That is, the given tetrahedra are supposed to form a triangulation and
    have GeodesicPiece's stored in tet.geodesic_pieces such that all endpoints
    of all pieces are at vertices of the tetrahedron, that is the line
    segment the GeodesicPiece represents is an edge of the tetrahedron.
    """
    mask, peripheral_base_subtet_indices = _tet_mask_and_peripheral_base_subtet_indices(tetrahedra)
    subtetrahedra = [Tetrahedron() if m else None for m in mask]
    _assign_orientations(subtetrahedra)
    for tet in tetrahedra:
        for i, perm in enumerate(Perm4.S4()):
            subtet_index = 24 * tet.Index + i
            subtet = subtetrahedra[subtet_index]
            if subtet is None:
                continue
            for face in range(3):
                other_perm = perm * _transpositions[face]
                j = _perm_to_index(other_perm)
                other_subtet_index = 24 * tet.Index + j
                if face == 1 and (not mask[other_subtet_index]):
                    other_perm = perm * Perm4((2, 1, 0, 3))
                    j = _perm_to_index(other_perm)
                    other_subtet_index = 24 * tet.Index + j
                if j > i:
                    subtet.attach(simplex.TwoSubsimplices[face], subtetrahedra[other_subtet_index], (0, 1, 2, 3))
            vertex = perm.image(simplex.V0)
            face = perm.image(simplex.F3)
            other_tet = tet.Neighbor[face]
            other_perm = tet.Gluing[face] * perm
            j = _perm_to_index(other_perm)
            other_subtet_index = 24 * other_tet.Index + j
            if other_subtet_index > subtet_index:
                subtet.attach(simplex.F3, subtetrahedra[other_subtet_index], (0, 1, 2, 3))
            subtet.post_drill_infos = {simplex.V0: tet.post_drill_infos[vertex], simplex.V1: CuspPostDrillInfo(), simplex.V2: CuspPostDrillInfo(), simplex.V3: CuspPostDrillInfo()}
            subtet.needs_peripheral_curves_fixed = False
            subtet.PeripheralCurves = [[{v: {f: 0 for f in simplex.TwoSubsimplices} for v in simplex.ZeroSubsimplices} for sheet in range(2)] for ml in range(2)]
            for ml in range(2):
                for sheet in range(2):
                    p = tet.PeripheralCurves[ml][sheet][vertex][face]
                    if p > 0 and perm.sign() == 0:
                        subtet.PeripheralCurves[ml][sheet][simplex.V0][simplex.F3] = p
                        subtet.needs_peripheral_curves_fixed = True
                    elif p < 0 and perm.sign() == 1:
                        subtet.PeripheralCurves[ml][1 - sheet][simplex.V0][simplex.F3] = p
    _fix_all_peripheral_curves(subtetrahedra)
    for i in peripheral_base_subtet_indices:
        install_peripheral_curves(subtetrahedra[i])
    return Mcomplex([subtet for s in [0, 1] for subtet in subtetrahedra if subtet and subtet.orientation == s])
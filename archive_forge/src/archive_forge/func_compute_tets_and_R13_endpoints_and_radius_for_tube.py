from ..drilling import compute_geodesic_info
from ..drilling.geodesic_tube import GeodesicTube
from ..drilling.line import distance_r13_lines
from ..snap.t3mlite import simplex # type: ignore
def compute_tets_and_R13_endpoints_and_radius_for_tube(self, radius):
    while True:
        safe_radius = self.dist_to_core_curve * 0.98
        if radius > safe_radius:
            radius = safe_radius
            break
        if self.geodesic_tube.covered_radius() > radius:
            break
        self.geodesic_tube._add_next_piece()
        piece = self.geodesic_tube.pieces[-1]
        for v in simplex.ZeroSubsimplices:
            core_curve = piece.tet.core_curves.get(v, None)
            if core_curve:
                d = distance_r13_lines(core_curve.r13_line, piece.lifted_geodesic)
                if d < self.dist_to_core_curve:
                    self.dist_to_core_curve = d
    result = []
    for piece in self.geodesic_tube.pieces:
        if piece.lower_bound > radius:
            break
        result.append((piece.tet.Index, [piece.tet.to_coordinates_in_symmetric_tet * pt for pt in piece.lifted_geodesic.points]))
    return (result, radius)
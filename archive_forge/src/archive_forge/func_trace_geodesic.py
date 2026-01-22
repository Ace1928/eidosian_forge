from .geodesic_info import GeodesicInfo
from .line import R13LineWithMatrix, distance_r13_lines
from . import constants
from . import epsilons
from . import exceptions
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..hyperboloid import r13_dot # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, List
def trace_geodesic(geodesic: GeodesicInfo, verified: bool):
    """
    Traces line segment through the tetrahedra in the hyperboloid
    model (though using time-like but not necessarily unit time-like vectors)
    starting from the given start point in the given tetrahdra
    to the given end point (assumed to be related to the start point
    by a primitive Decktransformation).

    The output is a (python) list of GeodesicPiece's (that is also
    a cyclic linked list). The first piece is going from the interior of a
    tetrahedron to a point on the face of the tetrahedron. The last piece
    goes the other way to close the loop. All other pieces go from a
    point on a face to a point on another face.

    If geodesic.line is set, it also checks that the geodesic is not
    too close to a core curve.
    """
    if geodesic.tet is None:
        raise ValueError('Expected geodesic with tetrahedron to start tracing.')
    start_point = geodesic.unnormalised_start_point
    direction = geodesic.unnormalised_end_point - geodesic.unnormalised_start_point
    line: Optional[R13LineWithMatrix] = geodesic.line
    tet: Tetrahedron = geodesic.tet
    face: int = simplex.T
    RF = start_point[0].parent()
    if verified:
        epsilon = 0
    else:
        epsilon = epsilons.compute_epsilon(RF)
    pieces: List[GeodesicPiece] = []
    param = RF(0)
    for i in range(constants.trace_max_steps):
        hit_face: Optional[int] = None
        hit_param = None
        for candidate_face, plane in tet.R13_unnormalised_planes.items():
            if candidate_face == face:
                continue
            candidate_param = compute_plane_intersection_param(plane, start_point, direction, verified)
            if candidate_param < param - epsilon:
                continue
            if not candidate_param > param + epsilon:
                raise InsufficientPrecisionError('When tracing the geodesic, the intersection with the next tetrahedron face was too close to the previous to tell them apart. Increasing the precision will probably avoid this problem.')
            if hit_param is None:
                hit_param = candidate_param
                hit_face = candidate_face
            elif candidate_param + epsilon < hit_param:
                hit_param = candidate_param
                hit_face = candidate_face
            elif not candidate_param > hit_param + epsilon:
                raise exceptions.RayHittingOneSkeletonError()
        if hit_param is None or hit_face is None:
            raise InsufficientPrecisionError('Could not find the next intersection of the geodesic with a tetrahedron face. Increasing the precision should solve this problem.')
        _verify_away_from_core_curve(line, tet, hit_face, epsilon)
        if hit_param > RF(1) + epsilon:
            hit_param = RF(1)
            T: int = simplex.T
            hit_face = T
        elif not hit_param < RF(1) - epsilon:
            raise InsufficientPrecisionError('Could not determine whether we finished tracing the geodesic. Increasing the precision will most likely fix the problem.')
        pieces.append(GeodesicPiece(geodesic.index, tet, [Endpoint(start_point + param * direction, face), Endpoint(start_point + hit_param * direction, hit_face)]))
        if hit_face == simplex.T:
            if tet is not geodesic.tet:
                raise InsufficientPrecisionError('Tracing geodesic ended up in a different tetrahedron than it started. Increasing the precision will probably fix this problem.')
            GeodesicPiece.make_linked_list(pieces)
            return pieces
        m = tet.O13_matrices[hit_face]
        start_point = m * start_point
        direction = m * direction
        if line is not None:
            line = line.transformed(m)
        param = hit_param
        face = tet.Gluing[hit_face].image(hit_face)
        tet = tet.Neighbor[hit_face]
    raise exceptions.UnfinishedTraceGeodesicError(constants.trace_max_steps)
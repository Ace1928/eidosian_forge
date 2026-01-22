from .geodesic_info import GeodesicInfo
from .line import R13LineWithMatrix, distance_r13_lines
from . import constants
from . import epsilons
from . import exceptions
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..hyperboloid import r13_dot # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, List
@staticmethod
def create_face_to_vertex_and_attach(index: int, tet: Tetrahedron, point: Endpoint, direction: int):
    """
        Creates a line segment between the given endpoint on
        a face and the opposite vertex. If direction is +1,
        the pieces goes from the endpoint to the vertex.
        If direction is -1, it goes the opposite way.

        Also appends the new geodesic piece to tet.geodesic_pieces.
        """
    if point.subsimplex not in simplex.TwoSubsimplices:
        raise ValueError('Expected point to be on a face, but its subsimplex is %d' % point.subsimplex)
    v = simplex.comp(point.subsimplex)
    return GeodesicPiece.create_and_attach(index, tet, [point, Endpoint(tet.R13_vertices[v], v)][::direction])
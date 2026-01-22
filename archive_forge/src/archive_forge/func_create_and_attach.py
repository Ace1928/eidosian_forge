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
def create_and_attach(index: int, tet: Tetrahedron, endpoints: Sequence[Endpoint]):
    """
        Creates a line segment and appends it to tet.geodesic_pieces.
        """
    g = GeodesicPiece(index, tet, endpoints)
    tet.geodesic_pieces.append(g)
    return g
from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron
from ..snap.t3mlite.mcomplex import VERBOSE
from .exceptions import GeneralPositionError
from .rational_linear_algebra import Vector3, QQ
from . import pl_utils
from . import stored_moves
from .mcomplex_with_expansion import McomplexWithExpansion
from .mcomplex_with_memory import McomplexWithMemory
from .barycentric_geometry import (BarycentricPoint, BarycentricArc,
import random
import collections
import time
def run_example_moves(self, k=None, straighten=True, push=True, round=True, careful=True, start=0):
    """
        Assuming this example appears in stored_moves, perform the
        first k moves.
        """
    db = stored_moves.move_db
    if self.name not in db:
        raise ValueError('Manifold not found in stored_moves')
    moves = stored_moves.move_db[self.name]['moves']
    if k is not None:
        moves = moves[start:k]
    self.perform_moves(moves, straighten, push, round, careful)
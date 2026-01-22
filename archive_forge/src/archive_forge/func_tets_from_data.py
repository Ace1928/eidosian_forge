from .simplex import *
from .tetrahedron import Tetrahedron
from .corner import Corner
from .arrow import Arrow
from .face import Face
from .edge import Edge
from .vertex import Vertex
from .surface import Surface, SpunSurface, ClosedSurface, ClosedSurfaceInCusped
from .perm4 import Perm4, inv
from . import files
from . import linalg
from . import homology
import sys
import random
import io
def tets_from_data(fake_tets):
    """
    Takes a list where the ith element represents the gluing data
    for the ith tetraherda::

      ( [Neighbors], [Glueings] )

    and creates the corresponding glued Tetraherda.
    """
    fake_tets = fake_tets
    num_tets = len(fake_tets)
    tets = [Tetrahedron() for i in range(num_tets)]
    for i in range(num_tets):
        neighbors, perms = fake_tets[i]
        for k in range(4):
            tets[i].attach(TwoSubsimplices[k], tets[neighbors[k]], perms[k])
    return tets
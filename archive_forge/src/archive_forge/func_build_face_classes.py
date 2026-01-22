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
def build_face_classes(self):
    """
        Construct the faces.
        """
    for tet in self.Tetrahedra:
        for two_subsimplex in TwoSubsimplices:
            if tet.Class[two_subsimplex] is None:
                newFace = Face()
                self.Faces.append(newFace)
                newFace.Corners.append(Corner(tet, two_subsimplex))
                tet.Class[two_subsimplex] = newFace
                othertet = tet.Neighbor[two_subsimplex]
                if othertet:
                    newFace.IntOrBdry = 'int'
                    othersubsimplex = tet.Gluing[two_subsimplex].image(two_subsimplex)
                    newFace.Corners.append(Corner(othertet, othersubsimplex))
                    othertet.Class[othersubsimplex] = newFace
                else:
                    newFace.IntOrBdry = 'bdry'
    for i in range(len(self.Faces)):
        self.Faces[i].Index = i
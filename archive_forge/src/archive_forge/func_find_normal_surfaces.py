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
def find_normal_surfaces(self, modp=0, print_progress=False, algorithm='FXrays'):
    """
        Convention is that the ordered quads are (Q03, Q13, Q23).
        """
    self.NormalSurfaces = []
    self.build_matrix()
    if algorithm == 'FXrays':
        try:
            import FXrays
        except ImportError:
            raise ImportError('You need to install the FXrays moduleif you want to find normal surfaces.')
        coeff_list = FXrays.find_Xrays(self.QuadMatrix.nrows(), self.QuadMatrix.ncols(), self.QuadMatrix.entries(), modp, print_progress=print_progress)
    elif algorithm == 'regina':
        T = self.regina_triangulation()
        import regina
        coeff_list = []
        tets = range(len(self))
        surfaces = regina.NNormalSurfaceList.enumerate(T, regina.NS_QUAD)
        for i in range(surfaces.getNumberOfSurfaces()):
            S = surfaces.getSurface(i)
            coeff_vector = [int(S.getQuadCoord(tet, quad).stringValue()) for tet in tets for quad in (2, 1, 0)]
            coeff_list.append(coeff_vector)
    else:
        raise ValueError("Algorithm must be in {'FXrays', 'regina'}")
    for coeff_vector in coeff_list:
        if max(self.LinkGenera) == 0:
            self.NormalSurfaces.append(ClosedSurface(self, coeff_vector))
        elif self.LinkGenera.count(1) == len(self.LinkGenera):
            self.NormalSurfaces.append(SpunSurface(self, coeff_vector))
        else:
            self.NormalSurfaces.append(Surface(self, coeff_vector))
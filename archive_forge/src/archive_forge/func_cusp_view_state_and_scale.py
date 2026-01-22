from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.snap.mcomplex_base import *
from snappy.verify.cuspCrossSection import *
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
from ..upper_halfspace.ideal_point import ideal_point_to_r13
from .hyperboloid_utilities import *
from .upper_halfspace_utilities import *
from .raytracing_data import *
from math import sqrt
def cusp_view_state_and_scale(self, which_cusp):
    vert = self.mcomplex.Vertices[which_cusp]
    corner = vert.Corners[0]
    tet = corner.Tetrahedron
    subsimplex = corner.Subsimplex
    area = self.areas[which_cusp]
    return (self.update_view_state((_cusp_view_matrix(tet, subsimplex, area), corner.Tetrahedron.Index, 0.0)), _cusp_view_scale(tet, subsimplex, area))
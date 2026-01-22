from .geodesic_tube_info import GeodesicTubeInfo
from .upper_halfspace_utilities import *
from ..drilling.geometric_structure import add_r13_geometry
from ..drilling.geodesic_tube import add_structures_necessary_for_tube
from ..snap.t3mlite import Mcomplex, simplex
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
def o13_matrix_taking_ideal_vertices_to_ideal_vertices(verts0, verts1):
    m1 = pgl2_matrix_taking_0_1_inf_to_given_points(*verts0)
    m2 = pgl2_matrix_taking_0_1_inf_to_given_points(*verts1)
    return pgl2c_to_o13(m2 * sl2c_inverse(m1))
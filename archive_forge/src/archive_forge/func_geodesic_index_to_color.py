from .geodesic_tube_info import GeodesicTubeInfo
from .upper_halfspace_utilities import *
from ..drilling.geometric_structure import add_r13_geometry
from ..drilling.geodesic_tube import add_structures_necessary_for_tube
from ..snap.t3mlite import Mcomplex, simplex
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
def geodesic_index_to_color(i):
    """
    Reimplementation of object_type_geodesic_tube case of
    material_params from fragment.glsl.
    """
    golden_angle_by_2_pi = 0.3819660112501051
    return hsv2rgb(golden_angle_by_2_pi * i + 0.1, 1.0, 1.0)
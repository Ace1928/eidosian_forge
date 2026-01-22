from .geodesic_tube_info import GeodesicTubeInfo
from .upper_halfspace_utilities import *
from ..drilling.geometric_structure import add_r13_geometry
from ..drilling.geodesic_tube import add_structures_necessary_for_tube
from ..snap.t3mlite import Mcomplex, simplex
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
def compute_geodesic_tube_info_key(geodesic_tube_info):
    l = geodesic_tube_info.complex_length
    return (int(l.real() * 100000.0), int(abs(l.imag() * 100000.0)), l.imag() > 1e-05, geodesic_tube_info.index)
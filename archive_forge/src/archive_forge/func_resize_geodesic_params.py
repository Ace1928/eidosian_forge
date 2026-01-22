from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
def resize_geodesic_params(self, enable=False):
    num = len(self.geodesics.geodesic_tube_infos) - len(self.ui_parameter_dict['geodesicTubeRadii'][1])
    self.ui_parameter_dict['geodesicTubeRadii'][1] += num * [0.02]
    self.ui_parameter_dict['geodesicTubeEnables'][1] += num * [enable]
from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
def update_geodesic_data_and_redraw(self):
    success = self._update_geodesic_data()
    self._update_shader()
    self.redraw_if_initialized()
    return success
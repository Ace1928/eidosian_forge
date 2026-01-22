from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
def setCameraParams(self, **kwds):
    valid_keys = {'center', 'rotation', 'distance', 'fov', 'elevation', 'azimuth'}
    if not valid_keys.issuperset(kwds):
        raise ValueError(f'valid keywords are {valid_keys}')
    self.setCameraPosition(pos=kwds.get('center'), distance=kwds.get('distance'), elevation=kwds.get('elevation'), azimuth=kwds.get('azimuth'), rotation=kwds.get('rotation'))
    if 'fov' in kwds:
        self.opts['fov'] = kwds['fov']
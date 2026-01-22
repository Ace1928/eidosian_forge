from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
def projectionMatrix(self, region=None):
    if region is None:
        region = (0, 0, self.deviceWidth(), self.deviceHeight())
    x0, y0, w, h = self.getViewport()
    dist = self.opts['distance']
    fov = self.opts['fov']
    nearClip = dist * 0.001
    farClip = dist * 1000.0
    r = nearClip * tan(0.5 * radians(fov))
    t = r * h / w
    left = r * ((region[0] - x0) * (2.0 / w) - 1)
    right = r * ((region[0] + region[2] - x0) * (2.0 / w) - 1)
    bottom = t * ((region[1] - y0) * (2.0 / h) - 1)
    top = t * ((region[1] + region[3] - y0) * (2.0 / h) - 1)
    tr = QtGui.QMatrix4x4()
    tr.frustum(left, right, bottom, top, nearClip, farClip)
    return tr
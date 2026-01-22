from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
def setProjection(self, region=None):
    m = self.projectionMatrix(region)
    glMatrixMode(GL_PROJECTION)
    glLoadMatrixf(np.array(m.data(), dtype=np.float32))
from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
def itemsAt(self, region=None):
    """
        Return a list of the items displayed in the region (x, y, w, h)
        relative to the widget.        
        """
    region = (region[0], self.deviceHeight() - (region[1] + region[3]), region[2], region[3])
    buf = glSelectBuffer(100000)
    try:
        glRenderMode(GL_SELECT)
        glInitNames()
        glPushName(0)
        self._itemNames = {}
        self.paintGL(region=region, useItemNames=True)
    finally:
        hits = glRenderMode(GL_RENDER)
    items = [(h.near, h.names[0]) for h in hits]
    items.sort(key=lambda i: i[0])
    return [self._itemNames[i[1]] for i in items]
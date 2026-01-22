from OpenGL.GL import *  # noqa
import numpy as np
from ...Qt import QtGui
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem
from ..MeshData import MeshData
def setShader(self, shader):
    """Set the shader used when rendering faces in the mesh. (see the GL shaders example)"""
    self.opts['shader'] = shader
    self.update()
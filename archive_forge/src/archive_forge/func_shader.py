from OpenGL.GL import *  # noqa
import numpy as np
from ...Qt import QtGui
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem
from ..MeshData import MeshData
def shader(self):
    shader = self.opts['shader']
    if isinstance(shader, shaders.ShaderProgram):
        return shader
    else:
        return shaders.getShaderProgram(shader)
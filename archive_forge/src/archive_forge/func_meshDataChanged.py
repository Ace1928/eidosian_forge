from OpenGL.GL import *  # noqa
import numpy as np
from ...Qt import QtGui
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem
from ..MeshData import MeshData
def meshDataChanged(self):
    """
        This method must be called to inform the item that the MeshData object
        has been altered.
        """
    self.vertexes = None
    self.faces = None
    self.normals = None
    self.colors = None
    self.edges = None
    self.edgeColors = None
    self.update()
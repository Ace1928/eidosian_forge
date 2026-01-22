from OpenGL.GL import *  # noqa
import numpy as np
from ...Qt import QtGui
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem
from ..MeshData import MeshData
def setMeshData(self, **kwds):
    """
        Set mesh data for this item. This can be invoked two ways:
        
        1. Specify *meshdata* argument with a new MeshData object
        2. Specify keyword arguments to be passed to MeshData(..) to create a new instance.
        """
    md = kwds.get('meshdata', None)
    if md is None:
        opts = {}
        for k in ['vertexes', 'faces', 'edges', 'vertexColors', 'faceColors']:
            try:
                opts[k] = kwds.pop(k)
            except KeyError:
                pass
        md = MeshData(**opts)
    self.opts['meshdata'] = md
    self.opts.update(kwds)
    self.meshDataChanged()
    self.update()
from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def resetTransform(self):
    """Reset this item's transform to an identity transformation."""
    self.__transform.setToIdentity()
    self.update()
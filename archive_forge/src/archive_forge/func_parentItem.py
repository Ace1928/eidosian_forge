from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def parentItem(self):
    """Return a this item's parent in the scenegraph hierarchy."""
    return self.__parent
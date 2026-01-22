from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def setVisible(self, vis):
    """Set the visibility of this item."""
    self.__visible = vis
    self.update()
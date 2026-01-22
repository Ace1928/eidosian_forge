from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def setParentItem(self, item):
    """Set this item's parent in the scenegraph hierarchy."""
    if self.__parent is not None:
        self.__parent.__children.remove(self)
    if item is not None:
        item.__children.add(self)
    self.__parent = item
    if self.__parent is not None and self.view() is not self.__parent.view():
        if self.view() is not None:
            self.view().removeItem(self)
        self.__parent.view().addItem(self)
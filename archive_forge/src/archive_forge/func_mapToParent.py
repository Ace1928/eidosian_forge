from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def mapToParent(self, point):
    tr = self.transform()
    if tr is None:
        return point
    return tr.map(point)
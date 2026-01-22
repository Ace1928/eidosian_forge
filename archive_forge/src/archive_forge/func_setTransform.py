from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def setTransform(self, tr):
    """Set the local transform for this object.

        Parameters
        ----------
        tr : pyqtgraph.Transform3D
            Tranformation from the local coordinate system to the parent's.
        """
    self.__transform = Transform3D(tr)
    self.update()
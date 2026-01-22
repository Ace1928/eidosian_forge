import sys
from PySide2.QtCore import(Property, QObject, QPropertyAnimation, Signal, Slot)
from PySide2.QtGui import (QGuiApplication, QMatrix4x4, QQuaternion, QVector3D, QWindow)
from PySide2.Qt3DCore import (Qt3DCore)
from PySide2.Qt3DRender import (Qt3DRender)
from PySide2.Qt3DExtras import (Qt3DExtras)
def setRadius(self, radius):
    if self._radius != radius:
        self._radius = radius
        self.updateMatrix()
        self.radiusChanged.emit()
from ..Qt import QtCore, QtGui, QtWidgets
def removeAllowedArea(self, area):
    self.allowedAreas.discard(area)
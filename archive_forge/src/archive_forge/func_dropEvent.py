from PySide2 import QtCore, QtGui, QtWidgets
import dragdroprobot_rc
def dropEvent(self, event):
    self.dragOver = False
    if event.mimeData().hasColor():
        self.color = QtGui.QColor(event.mimeData().colorData())
    elif event.mimeData().hasImage():
        self.pixmap = QtGui.QPixmap(event.mimeData().imageData())
    self.update()
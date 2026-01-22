from .. import functions as functions
from ..Qt import QtCore, QtGui, QtWidgets
def dialogColorChanged(self, color):
    if color.isValid():
        self.setColor(color, finished=False)
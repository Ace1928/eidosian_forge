from .. import functions as functions
from ..Qt import QtCore, QtGui, QtWidgets
def colorRejected(self):
    self.setColor(self.origColor, finished=False)
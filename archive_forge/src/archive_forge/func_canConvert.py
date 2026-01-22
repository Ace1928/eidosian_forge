import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def canConvert(self, mime, flav):
    if self.mimeFor(flav) == mime:
        return True
    else:
        return False
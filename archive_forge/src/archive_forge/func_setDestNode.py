import sys
import weakref
import math
from PySide2 import QtCore, QtGui, QtWidgets
def setDestNode(self, node):
    self.dest = weakref.ref(node)
    self.adjust()
import inspect
import weakref
from .Qt import QtCore, QtWidgets
def findWidget(self, name):
    for w in self.widgetList:
        if self.widgetList[w] == name:
            return w
    return None
from ..Qt import QtCore, QtWidgets
def childItems(self):
    return [self.child(i) for i in range(self.childCount())]
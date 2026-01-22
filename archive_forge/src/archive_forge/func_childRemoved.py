from ..Qt import QtCore, QtGui, QtWidgets
def childRemoved(self, param, child):
    for i in range(self.childCount()):
        item = self.child(i)
        if item.param is child:
            self.takeChild(i)
            break
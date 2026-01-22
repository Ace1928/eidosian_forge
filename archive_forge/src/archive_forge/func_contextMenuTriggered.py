from ..Qt import QtCore, QtGui, QtWidgets
def contextMenuTriggered(self, name):

    def trigger():
        self.param.contextMenu(name)
    return trigger
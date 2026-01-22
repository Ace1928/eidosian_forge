from PySide2 import QtCore, QtGui, QtWidgets
def doLayout(self, rect, testOnly):
    x = rect.x()
    y = rect.y()
    lineHeight = 0
    for item in self.itemList:
        wid = item.widget()
        spaceX = self.spacing() + wid.style().layoutSpacing(QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Horizontal)
        spaceY = self.spacing() + wid.style().layoutSpacing(QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Vertical)
        nextX = x + item.sizeHint().width() + spaceX
        if nextX - spaceX > rect.right() and lineHeight > 0:
            x = rect.x()
            y = y + lineHeight + spaceY
            nextX = x + item.sizeHint().width() + spaceX
            lineHeight = 0
        if not testOnly:
            item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))
        x = nextX
        lineHeight = max(lineHeight, item.sizeHint().height())
    return y + lineHeight - rect.y()
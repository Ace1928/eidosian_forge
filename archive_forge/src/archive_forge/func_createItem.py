from PySide2 import QtCore, QtGui, QtWidgets
def createItem(minimum, preferred, maximum, name):
    w = QtWidgets.QGraphicsProxyWidget()
    w.setWidget(QtWidgets.QPushButton(name))
    w.setMinimumSize(minimum)
    w.setPreferredSize(preferred)
    w.setMaximumSize(maximum)
    w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    return w
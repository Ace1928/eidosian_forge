from PySide2 import QtCore, QtGui, QtWidgets
def setSourceModel(self, model):
    self.proxyModel.setSourceModel(model)
    self.sourceView.setModel(model)
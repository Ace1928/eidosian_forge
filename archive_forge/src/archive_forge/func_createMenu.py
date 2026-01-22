from PySide2 import QtCore, QtGui, QtWidgets
def createMenu(self):
    self.menuBar = QtWidgets.QMenuBar()
    self.fileMenu = QtWidgets.QMenu('&File', self)
    self.exitAction = self.fileMenu.addAction('E&xit')
    self.menuBar.addMenu(self.fileMenu)
    self.exitAction.triggered.connect(self.accept)
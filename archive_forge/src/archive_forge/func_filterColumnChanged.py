from PySide2 import QtCore, QtGui, QtWidgets
def filterColumnChanged(self):
    self.proxyModel.setFilterKeyColumn(self.filterColumnComboBox.currentIndex())
from PySide2 import QtCore, QtGui, QtWidgets
def createFilesTable(self):
    self.filesTable = QtWidgets.QTableWidget(0, 2)
    self.filesTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    self.filesTable.setHorizontalHeaderLabels(('File Name', 'Size'))
    self.filesTable.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
    self.filesTable.verticalHeader().hide()
    self.filesTable.setShowGrid(False)
    self.filesTable.cellActivated.connect(self.openFileOfItem)
from PySide2 import QtCore, QtGui, QtWidgets
def createGridGroupBox(self):
    self.gridGroupBox = QtWidgets.QGroupBox('Grid layout')
    layout = QtWidgets.QGridLayout()
    for i in range(Dialog.NumGridRows):
        label = QtWidgets.QLabel('Line %d:' % (i + 1))
        lineEdit = QtWidgets.QLineEdit()
        layout.addWidget(label, i + 1, 0)
        layout.addWidget(lineEdit, i + 1, 1)
    self.smallEditor = QtWidgets.QTextEdit()
    self.smallEditor.setPlainText('This widget takes up about two thirds of the grid layout.')
    layout.addWidget(self.smallEditor, 0, 2, 4, 1)
    layout.setColumnStretch(1, 10)
    layout.setColumnStretch(2, 20)
    self.gridGroupBox.setLayout(layout)
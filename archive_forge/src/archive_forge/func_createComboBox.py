from PySide2 import QtCore, QtGui, QtWidgets
def createComboBox(self, text=''):
    comboBox = QtWidgets.QComboBox()
    comboBox.setEditable(True)
    comboBox.addItem(text)
    comboBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    return comboBox
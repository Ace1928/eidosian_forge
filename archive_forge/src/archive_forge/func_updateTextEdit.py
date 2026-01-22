from PySide2 import QtCore, QtGui, QtWidgets
def updateTextEdit(self):
    mib = self.encodingComboBox.itemData(self.encodingComboBox.currentIndex())
    codec = QtCore.QTextCodec.codecForMib(mib)
    data = QtCore.QTextStream(self.encodedData)
    data.setAutoDetectUnicode(False)
    data.setCodec(codec)
    self.decodedStr = data.readAll()
    self.textEdit.setPlainText(self.decodedStr)
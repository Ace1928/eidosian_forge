from PySide2 import QtCore, QtGui, QtWidgets
def filterRegExpChanged(self):
    syntax_nr = self.filterSyntaxComboBox.itemData(self.filterSyntaxComboBox.currentIndex())
    syntax = QtCore.QRegExp.PatternSyntax(syntax_nr)
    if self.filterCaseSensitivityCheckBox.isChecked():
        caseSensitivity = QtCore.Qt.CaseSensitive
    else:
        caseSensitivity = QtCore.Qt.CaseInsensitive
    regExp = QtCore.QRegExp(self.filterPatternLineEdit.text(), caseSensitivity, syntax)
    self.proxyModel.setFilterRegExp(regExp)
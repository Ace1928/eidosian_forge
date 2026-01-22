from ..Qt import QtCore, QtWidgets
def insertTopLevelItem(self, index, item):
    QtWidgets.QTreeWidget.insertTopLevelItem(self, index, item)
    self.informTreeWidgetChange(item)
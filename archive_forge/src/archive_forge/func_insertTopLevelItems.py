from ..Qt import QtCore, QtWidgets
def insertTopLevelItems(self, index, items):
    QtWidgets.QTreeWidget.insertTopLevelItems(self, index, items)
    for item in items:
        self.informTreeWidgetChange(item)
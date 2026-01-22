from ..Qt import QtCore, QtWidgets
def takeChildren(self):
    childs = self._real_item.takeChildren()
    for child in childs:
        TreeWidget.informTreeWidgetChange(child)
    return childs
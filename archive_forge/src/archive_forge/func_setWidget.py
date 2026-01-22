from ..Qt import QtCore, QtWidgets
def setWidget(self, column, widget):
    if column in self._widgets:
        self.removeWidget(column)
    self._widgets[column] = widget
    tree = self.treeWidget()
    if tree is None:
        return
    else:
        tree.setItemWidget(self, column, widget)
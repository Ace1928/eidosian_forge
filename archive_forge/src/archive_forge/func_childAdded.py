from ..Qt import QtCore, QtGui, QtWidgets
def childAdded(self, param, child, pos):
    item = child.makeTreeItem(depth=self.depth + 1)
    self.insertChild(pos, item)
    item.treeWidgetChanged()
    for i, ch in enumerate(child):
        item.childAdded(child, ch, i)
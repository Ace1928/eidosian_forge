from ..Qt import QtCore, QtWidgets
def setItemWidget(self, item, col, wid):
    """
        Overrides QTreeWidget.setItemWidget such that widgets are added inside an invisible wrapper widget.
        This makes it possible to move the item in and out of the tree without its widgets being automatically deleted.
        """
    w = QtWidgets.QWidget()
    l = QtWidgets.QVBoxLayout()
    l.setContentsMargins(0, 0, 0, 0)
    w.setLayout(l)
    w.setSizePolicy(wid.sizePolicy())
    w.setMinimumHeight(wid.minimumHeight())
    w.setMinimumWidth(wid.minimumWidth())
    l.addWidget(wid)
    w.realChild = wid
    self.placeholders.append(w)
    QtWidgets.QTreeWidget.setItemWidget(self, item, col, w)
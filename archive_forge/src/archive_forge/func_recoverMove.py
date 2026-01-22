from ..Qt import QtCore, QtWidgets
def recoverMove(self, item):
    for i in range(self.columnCount()):
        w = item.__widgets[i]
        if w is None:
            continue
        self.setItemWidget(item, i, w)
    for i in range(item.childCount()):
        self.recoverMove(item.child(i))
    item.setExpanded(False)
    QtWidgets.QApplication.instance().processEvents()
    item.setExpanded(item.__expanded)
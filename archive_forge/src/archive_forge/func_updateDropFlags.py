from ..Qt import QtCore, QtWidgets
def updateDropFlags(self):
    if self.childNestingLimit is None:
        pass
    else:
        items = self.listAllItems()
        for item in items:
            parentCount = 0
            p = item.parent()
            while p is not None:
                parentCount += 1
                p = p.parent()
            if parentCount >= self.childNestingLimit:
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled)
            else:
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsDropEnabled)
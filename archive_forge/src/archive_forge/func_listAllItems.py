from ..Qt import QtCore, QtWidgets
def listAllItems(self, item=None):
    items = []
    if item is not None:
        items.append(item)
    else:
        item = self.invisibleRootItem()
    for cindex in range(item.childCount()):
        foundItems = self.listAllItems(item=item.child(cindex))
        for f in foundItems:
            items.append(f)
    return items
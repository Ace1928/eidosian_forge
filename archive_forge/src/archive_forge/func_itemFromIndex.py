from ..Qt import QtCore, QtWidgets
def itemFromIndex(self, index):
    """Return the item and column corresponding to a QModelIndex.
        """
    col = index.column()
    rows = []
    while index.row() >= 0:
        rows.insert(0, index.row())
        index = index.parent()
    item = self.topLevelItem(rows[0])
    for row in rows[1:]:
        item = item.child(row)
    return (item, col)
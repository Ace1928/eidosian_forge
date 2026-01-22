import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
@_defersort
def setRow(self, row, vals):
    if row > self.rowCount() - 1:
        self.setRowCount(row + 1)
    for col in range(len(vals)):
        val = vals[col]
        item = self.itemClass(val, row)
        item.setEditable(self.editable)
        sortMode = self.sortModes.get(col, None)
        if sortMode is not None:
            item.setSortMode(sortMode)
        format = self._formats.get(col, self._formats[None])
        item.setFormat(format)
        self.items.append(item)
        self.setItem(row, col, item)
        item.setValue(val)
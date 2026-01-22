import sys, traceback
from ..Qt import QtWidgets, QtGui
def selectedFrame(self):
    """Return the currently selected stack frame (or None if there is no selection)
        """
    sel = self.selectedItems()
    if len(sel) == 0:
        return None
    else:
        return sel[0].frame
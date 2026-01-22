from ..Qt import QtWidgets
def nextRow(self):
    """Advance to next row for automatic widget placement"""
    self.currentRow += 1
    self.currentCol = 0
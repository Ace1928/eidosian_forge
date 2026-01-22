from ..Qt import QtWidgets
def getWidget(self, row, col):
    """Return the widget in (*row*, *col*)"""
    return self.rows[row][col]
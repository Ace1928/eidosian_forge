from PySide2.QtCore import (Qt, QAbstractTableModel, QModelIndex)
def columnCount(self, index=QModelIndex()):
    """ Returns the number of columns the model holds. """
    return 2
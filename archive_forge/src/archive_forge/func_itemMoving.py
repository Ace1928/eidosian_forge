from ..Qt import QtCore, QtWidgets
def itemMoving(self, item, parent, index):
    """Called when item has been dropped elsewhere in the tree.
        Return True to accept the move, False to reject."""
    return True
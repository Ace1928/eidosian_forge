from .parameterTypes import GroupParameterItem
from ..Qt import QtCore, QtWidgets, mkQApp
from ..widgets.TreeWidget import TreeWidget
from .ParameterItem import ParameterItem
def nextFocusableChild(self, root, startItem=None, forward=True):
    if startItem is None:
        if forward:
            index = 0
        else:
            index = root.childCount() - 1
    elif forward:
        index = root.indexOfChild(startItem) + 1
    else:
        index = root.indexOfChild(startItem) - 1
    if forward:
        inds = list(range(index, root.childCount()))
    else:
        inds = list(range(index, -1, -1))
    for i in inds:
        item = root.child(i)
        if hasattr(item, 'isFocusable') and item.isFocusable():
            return item
        else:
            item = self.nextFocusableChild(item, forward=forward)
            if item is not None:
                return item
    return None
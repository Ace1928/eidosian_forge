from .parameterTypes import GroupParameterItem
from ..Qt import QtCore, QtWidgets, mkQApp
from ..widgets.TreeWidget import TreeWidget
from .ParameterItem import ParameterItem
def itemExpandedEvent(self, item):
    if hasattr(item, 'expandedChangedEvent'):
        item.expandedChangedEvent(True)
from .parameterTypes import GroupParameterItem
from ..Qt import QtCore, QtWidgets, mkQApp
from ..widgets.TreeWidget import TreeWidget
from .ParameterItem import ParameterItem
def updatePalette(self):
    """
        called when application palette changes
        This should ensure that the color theme of the OS is applied to the GroupParameterItems
        should the theme chang while the application is running.
        """
    for item in self.listAllItems():
        if isinstance(item, GroupParameterItem):
            item.updateDepth(item.depth)
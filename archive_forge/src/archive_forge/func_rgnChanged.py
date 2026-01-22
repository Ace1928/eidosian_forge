import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
def rgnChanged(self, item):
    region = item.getRegion()
    self.stateGroup.setState({'start': region[0], 'stop': region[1]})
    self.update()
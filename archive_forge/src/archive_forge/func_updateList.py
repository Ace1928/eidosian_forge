import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
def updateList(self, data):
    if hasattr(data, 'implements') and data.implements('MetaArray'):
        cols = data.listColumns()
        for ax in cols:
            if len(cols[ax]) > 0:
                self.axis = ax
                cols = set(cols[ax])
                break
    else:
        cols = list(data.dtype.fields.keys())
    rem = set()
    for c in self.columns:
        if c not in cols:
            self.removeTerminal(c)
            rem.add(c)
    self.columns -= rem
    self.columnList.blockSignals(True)
    self.columnList.clear()
    for c in cols:
        item = QtWidgets.QListWidgetItem(c)
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        if c in self.columns:
            item.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.columnList.addItem(item)
    self.columnList.blockSignals(False)
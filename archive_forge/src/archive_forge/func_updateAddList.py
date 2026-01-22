import builtins
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
def updateAddList(self):
    self.addWidget.blockSignals(True)
    try:
        self.addWidget.clear()
        self.addWidget.addItem(self.param.opts['addText'])
        for t in self.param.opts['addList']:
            self.addWidget.addItem(t)
    finally:
        self.addWidget.blockSignals(False)
import builtins
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
def widgetEventFilter(self, obj, ev):
    if ev.type() == ev.Type.KeyPress:
        if ev.key() == QtCore.Qt.Key.Key_Tab:
            self.focusNext(forward=True)
            return True
        elif ev.key() == QtCore.Qt.Key.Key_Backtab:
            self.focusNext(forward=False)
            return True
    return False
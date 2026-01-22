import numpy as np
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from .basetypes import Emitter, WidgetParameterItem
def onMove(pos):
    self.sigChanging.emit(self, self.span[pos].item())
import builtins
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
def defaultClicked(self):
    self.param.setToDefault()
    self.updateDefaultBtn()
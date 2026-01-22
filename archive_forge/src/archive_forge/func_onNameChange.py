from ...Qt import QtCore, QtWidgets, QtGui
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
def onNameChange(self, param, name):
    self.updateOpts(param, dict(title=param.title()))
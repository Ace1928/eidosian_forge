from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
def updateDefaultBtn(self):
    self.metaBtns['default'].setEnabled(not self.param.valueIsDefault() and self.param.opts['enabled'] and self.param.writable())
    return
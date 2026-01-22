from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
def updateLimits(self, _param, limits):
    oldOpts = self.names
    val = self.opts.get('value', None)
    self.blockTreeChangeSignal()
    self.clearChildren()
    self.forward, self.reverse = ListParameter.mapping(limits)
    if self.opts.get('exclusive'):
        typ = 'radio'
    else:
        typ = 'bool'
    for chName in self.forward:
        newVal = bool(oldOpts.get(chName, False))
        child = BoolOrRadioParameter(type=typ, name=chName, value=newVal, default=None)
        self.addChild(child)
        child.blockTreeChangeSignal()
        child.sigValueChanged.connect(self._onChildChanging)
    self.treeStateChanges.clear()
    self.unblockTreeChangeSignal()
    self.setValue(val)
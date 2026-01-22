import re
from contextlib import ExitStack
from ... import functions as fn
from ...Qt import QtCore, QtWidgets
from ...SignalProxy import SignalProxy
from ...widgets.PenPreviewLabel import PenPreviewLabel
from . import GroupParameterItem, WidgetParameterItem
from .basetypes import GroupParameter, Parameter, ParameterItem
from .qtenum import QtEnumParameter
def mkPen(self, *args, **kwargs):
    """Thin wrapper around fn.mkPen which accepts the serialized state from saveState"""
    if len(args) == 1 and isinstance(args[0], tuple) and (len(args[0]) == len(self.childs)):
        opts = dict(zip(self.names, args[0]))
        self.applyOptsToPen(**opts)
        args = (self.pen,)
        kwargs = {}
    return fn.mkPen(*args, **kwargs)
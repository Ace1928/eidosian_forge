import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def emitStateChanged(self, changeDesc, data):
    self.sigStateChanged.emit(self, changeDesc, data)
    self.treeStateChanges.append((self, changeDesc, data))
    self.emitTreeChanges()
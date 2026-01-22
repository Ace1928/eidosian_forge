import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def blockTreeChangeSignal(self):
    """
        Used to temporarily block and accumulate tree change signals.
        *You must remember to unblock*, so it is advisable to use treeChangeBlocker() instead.
        """
    self.blockTreeChangeEmit += 1
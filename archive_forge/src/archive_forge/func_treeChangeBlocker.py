import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def treeChangeBlocker(self):
    """
        Return an object that can be used to temporarily block and accumulate
        sigTreeStateChanged signals. This is meant to be used when numerous changes are 
        about to be made to the tree and only one change signal should be
        emitted at the end.
        
        Example::

            with param.treeChangeBlocker():
                param.addChild(...)
                param.removeChild(...)
                param.setValue(...)
        """
    return SignalBlocker(self.blockTreeChangeSignal, self.unblockTreeChangeSignal)
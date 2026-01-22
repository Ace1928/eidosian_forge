from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
def filterChanged(self):
    self.sigFilterChanged.emit(self)
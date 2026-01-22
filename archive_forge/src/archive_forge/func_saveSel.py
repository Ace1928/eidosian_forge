import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
def saveSel(self):
    """Save selected data to file."""
    self.save(self.serialize(useSelection=True))
import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def setSingleStep(self, step):
    """Set the step size used when responding to the mouse wheel, arrow
        buttons, or arrow keys.
        """
    self.setOpts(step=step)
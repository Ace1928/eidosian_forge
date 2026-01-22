import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
def raiseDock(self):
    """If this Dock is stacked underneath others, raise it to the top."""
    self.container().raiseDock(self)
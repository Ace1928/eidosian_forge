import numpy as np
from .. import Qt, colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def getLevels(self):
    """
        Returns a tuple containing the current level settings. See :func:`~setLevels`.
        The format is ``(low, high)``.
        """
    return self.levels
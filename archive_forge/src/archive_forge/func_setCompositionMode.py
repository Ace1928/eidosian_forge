from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def setCompositionMode(self, mode):
    """
        Change the composition mode of the item. This is useful when overlaying
        multiple items.
        
        Parameters
        ----------
        mode : ``QtGui.QPainter.CompositionMode``
            Composition of the item, often used when overlaying items.  Common
            options include:

            ``QPainter.CompositionMode.CompositionMode_SourceOver`` (Default)
            Image replaces the background if it is opaque. Otherwise, it uses
            the alpha channel to blend the image with the background.

            ``QPainter.CompositionMode.CompositionMode_Overlay`` Image color is
            mixed with the background color to reflect the lightness or
            darkness of the background

            ``QPainter.CompositionMode.CompositionMode_Plus`` Both the alpha
            and color of the image and background pixels are added together.

            ``QPainter.CompositionMode.CompositionMode_Plus`` The output is the
            image color multiplied by the background.

            See ``QPainter::CompositionMode`` in the Qt Documentation for more
            options and details
        """
    self.opts['compositionMode'] = mode
    self.update()
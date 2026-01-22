import warnings
from collections.abc import Callable
import numpy
from .. import colormap
from .. import debug as debug
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..util.cupy_helper import getCupy
from .GraphicsObject import GraphicsObject
def setDrawKernel(self, kernel=None, mask=None, center=(0, 0), mode='set'):
    self.drawKernel = kernel
    self.drawKernelCenter = center
    self.drawMode = mode
    self.drawMask = mask
import weakref
from math import atan2, degrees
from ..functions import clip_scalar
from ..Qt import QtCore, QtWidgets
from . import ArrowItem
from .GraphicsObject import GraphicsObject
def setIndex(self, index):
    self.setProperty('index', int(index))
from math import atan2, degrees
import numpy as np
from . import SRTTransform
from .Qt import QtGui
from .Transform3D import Transform3D
from .Vector import Vector
def getTranslation(self):
    return Vector(self._state['pos'])
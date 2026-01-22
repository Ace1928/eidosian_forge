import os
from math import log10
from time import perf_counter
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..graphicsItems.GradientEditorItem import addGradientListToDocstring
from ..graphicsItems.ImageItem import ImageItem
from ..graphicsItems.InfiniteLine import InfiniteLine
from ..graphicsItems.LinearRegionItem import LinearRegionItem
from ..graphicsItems.ROI import ROI
from ..graphicsItems.ViewBox import ViewBox
from ..graphicsItems.VTickGroup import VTickGroup
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
from . import ImageViewTemplate_generic as ui_template
def evalKeyState(self):
    if len(self.keysPressed) == 1:
        key = list(self.keysPressed.keys())[0]
        if key == QtCore.Qt.Key.Key_Right:
            self.play(20)
            self.jumpFrames(1)
            self.lastPlayTime = perf_counter() + 0.2
        elif key == QtCore.Qt.Key.Key_Left:
            self.play(-20)
            self.jumpFrames(-1)
            self.lastPlayTime = perf_counter() + 0.2
        elif key == QtCore.Qt.Key.Key_Up:
            self.play(-100)
        elif key == QtCore.Qt.Key.Key_Down:
            self.play(100)
        elif key == QtCore.Qt.Key.Key_PageUp:
            self.play(-1000)
        elif key == QtCore.Qt.Key.Key_PageDown:
            self.play(1000)
    else:
        self.play(0)
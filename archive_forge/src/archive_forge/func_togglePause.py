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
def togglePause(self):
    if self.playTimer.isActive():
        self.play(0)
    elif self.playRate == 0:
        if self._pausedPlayRate is not None:
            fps = self._pausedPlayRate
        else:
            fps = (self.nframes() - 1) / (self.tVals[-1] - self.tVals[0])
        self.play(fps)
    else:
        self.play(self.playRate)
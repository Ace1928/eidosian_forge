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
def timeIndex(self, slider):
    """
        Returns
        -------
        int
            The index of the frame closest to the timeline slider.
        float
            The time value of the slider.
        """
    if not self.hasTimeAxis():
        return (0, 0.0)
    t = slider.value()
    xv = self.tVals
    if xv is None:
        ind = int(t)
    else:
        if len(xv) < 2:
            return (0, 0.0)
        inds = np.argwhere(xv <= t)
        if len(inds) < 1:
            return (0, t)
        ind = inds[-1, 0]
    return (ind, t)
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
def buildMenu(self):
    self.menu = QtWidgets.QMenu()
    self.normAction = QtGui.QAction(translate('ImageView', 'Normalization'), self.menu)
    self.normAction.setCheckable(True)
    self.normAction.toggled.connect(self.normToggled)
    self.menu.addAction(self.normAction)
    self.exportAction = QtGui.QAction(translate('ImageView', 'Export'), self.menu)
    self.exportAction.triggered.connect(self.exportClicked)
    self.menu.addAction(self.exportAction)
import collections.abc
import os
import warnings
import weakref
import numpy as np
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.FileDialog import FileDialog
from ..AxisItem import AxisItem
from ..ButtonItem import ButtonItem
from ..GraphicsWidget import GraphicsWidget
from ..InfiniteLine import InfiniteLine
from ..LabelItem import LabelItem
from ..LegendItem import LegendItem
from ..PlotCurveItem import PlotCurveItem
from ..PlotDataItem import PlotDataItem
from ..ScatterPlotItem import ScatterPlotItem
from ..ViewBox import ViewBox
from . import plotConfigTemplate_generic as ui_template
def updateDecimation(self):
    """
        Reduce or increase number of visible curves according to value set by the `Max Traces` spinner,
        if `Max Traces` is checked in the context menu. Destroy curves that are not visible if 
        `forget traces` is checked. In most cases, this function is called automaticaly when the 
        `Max Traces` GUI elements are triggered. It is also alled when the state of PlotItem is updated,
        its state is restored, or new items added added/removed.
        
        This can cause an unexpected or conflicting state of curve visibility (or destruction) if curve
        visibilities are controlled externally. In the case of external control it is advised to disable
        the `Max Traces` checkbox (or context menu) to prevent unexpected curve state changes.
        """
    if not self.ctrl.maxTracesCheck.isChecked():
        return
    else:
        numCurves = self.ctrl.maxTracesSpin.value()
    if self.ctrl.forgetTracesCheck.isChecked():
        for curve in self.curves[:-numCurves]:
            curve.clear()
            self.removeItem(curve)
    for i, curve in enumerate(reversed(self.curves)):
        if i < numCurves:
            curve.show()
        else:
            curve.hide()
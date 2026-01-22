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
def setContextMenuActionVisible(self, name: str, visible: bool) -> None:
    """
        Changes the context menu action visibility

        Parameters
        ----------
        name: str
            Action name
            must be one of 'Transforms', 'Downsample', 'Average','Alpha', 'Grid', or 'Points'
        visible: bool
            Determines if action will be display
            True action is visible
            False action is invisible.
        """
    translated_name = translate('PlotItem', name)
    for action in self.ctrlMenu.actions():
        if action.text() == translated_name:
            action.setVisible(visible)
            break
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
def writeCsv(self, fileName=None):
    if fileName is None:
        self._chooseFilenameDialog(handler=self.writeCsv)
        return
    fileName = str(fileName)
    PlotItem.lastFileDir = os.path.dirname(fileName)
    data = [c.getData() for c in self.curves]
    with open(fileName, 'w') as fd:
        i = 0
        while True:
            done = True
            for d in data:
                if i < len(d[0]):
                    fd.write('%g,%g,' % (d[0][i], d[1][i]))
                    done = False
                else:
                    fd.write(' , ,')
            fd.write('\n')
            if done:
                break
            i += 1
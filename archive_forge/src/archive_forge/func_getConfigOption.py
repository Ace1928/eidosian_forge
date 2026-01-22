import importlib
import os
import sys
import numpy  # # pyqtgraph requires numpy
from .colors import palette
from .Qt import QtCore, QtGui, QtWidgets
from .Qt import exec_ as exec
from .Qt import mkQApp
import atexit
from .colormap import *
from .functions import *
from .graphicsItems.ArrowItem import *
from .graphicsItems.AxisItem import *
from .graphicsItems.BarGraphItem import *
from .graphicsItems.ButtonItem import *
from .graphicsItems.ColorBarItem import *
from .graphicsItems.CurvePoint import *
from .graphicsItems.DateAxisItem import *
from .graphicsItems.ErrorBarItem import *
from .graphicsItems.FillBetweenItem import *
from .graphicsItems.GradientEditorItem import *
from .graphicsItems.GradientLegend import *
from .graphicsItems.GraphicsItem import *
from .graphicsItems.GraphicsLayout import *
from .graphicsItems.GraphicsObject import *
from .graphicsItems.GraphicsWidget import *
from .graphicsItems.GraphicsWidgetAnchor import *
from .graphicsItems.GraphItem import *
from .graphicsItems.GridItem import *
from .graphicsItems.HistogramLUTItem import *
from .graphicsItems.ImageItem import *
from .graphicsItems.InfiniteLine import *
from .graphicsItems.IsocurveItem import *
from .graphicsItems.ItemGroup import *
from .graphicsItems.LabelItem import *
from .graphicsItems.LegendItem import *
from .graphicsItems.LinearRegionItem import *
from .graphicsItems.MultiPlotItem import *
from .graphicsItems.PColorMeshItem import *
from .graphicsItems.PlotCurveItem import *
from .graphicsItems.PlotDataItem import *
from .graphicsItems.PlotItem import *
from .graphicsItems.ROI import *
from .graphicsItems.ScaleBar import *
from .graphicsItems.ScatterPlotItem import *
from .graphicsItems.TargetItem import *
from .graphicsItems.TextItem import *
from .graphicsItems.UIGraphicsItem import *
from .graphicsItems.ViewBox import *
from .graphicsItems.VTickGroup import *
from .GraphicsScene import GraphicsScene
from .imageview import *
from .metaarray import MetaArray
from .Point import Point
from .Qt import isQObjectAlive
from .SignalProxy import *
from .SRTTransform import SRTTransform
from .SRTTransform3D import SRTTransform3D
from .ThreadsafeTimer import *
from .Transform3D import Transform3D
from .util.cupy_helper import getCupy
from .Vector import Vector
from .WidgetGroup import *
from .widgets.BusyCursor import *
from .widgets.CheckTable import *
from .widgets.ColorButton import *
from .widgets.ColorMapWidget import *
from .widgets.ComboBox import *
from .widgets.DataFilterWidget import *
from .widgets.DataTreeWidget import *
from .widgets.DiffTreeWidget import *
from .widgets.FeedbackButton import *
from .widgets.FileDialog import *
from .widgets.GradientWidget import *
from .widgets.GraphicsLayoutWidget import *
from .widgets.GraphicsView import *
from .widgets.GroupBox import GroupBox
from .widgets.HistogramLUTWidget import *
from .widgets.JoystickButton import *
from .widgets.LayoutWidget import *
from .widgets.MultiPlotWidget import *
from .widgets.PathButton import *
from .widgets.PlotWidget import *
from .widgets.ProgressDialog import *
from .widgets.RawImageWidget import *
from .widgets.RemoteGraphicsView import RemoteGraphicsView
from .widgets.ScatterPlotWidget import *
from .widgets.SpinBox import *
from .widgets.TableWidget import *
from .widgets.TreeWidget import *
from .widgets.ValueLabel import *
from .widgets.VerticalLabel import *
def getConfigOption(opt):
    """Return the value of a single global configuration option.
    """
    return CONFIG_OPTIONS[opt]
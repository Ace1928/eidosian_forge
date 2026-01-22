from unittest.mock import Mock
import numpy as np
import panel as pn
from bokeh.document import Document
from pyviz_comms import Comm
import holoviews as hv
from holoviews.streams import (
from .test_plot import TestPlotlyPlot
def build_scatter(scale):
    return hv.Scatter(ys * scale)
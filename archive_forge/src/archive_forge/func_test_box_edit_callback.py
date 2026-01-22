import datetime as dt
from collections import deque, namedtuple
from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
import pyviz_comms as comms
from bokeh.events import Tap
from bokeh.io.doc import set_curdoc
from bokeh.models import ColumnDataSource, Plot, PolyEditTool, Range1d, Selection
from holoviews.core import DynamicMap
from holoviews.core.options import Store
from holoviews.element import Box, Curve, Points, Polygons, Rectangles, Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import (
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import (
def test_box_edit_callback(self):
    boxes = Rectangles([(-0.5, -0.5, 0.5, 0.5)])
    box_edit = BoxEdit(source=boxes)
    plot = bokeh_server_renderer.get_plot(boxes)
    self.assertIsInstance(plot.callbacks[0], BoxEditCallback)
    callback = plot.callbacks[0]
    source = plot.handles['cds']
    self.assertEqual(source.data['left'], [-0.5])
    self.assertEqual(source.data['bottom'], [-0.5])
    self.assertEqual(source.data['right'], [0.5])
    self.assertEqual(source.data['top'], [0.5])
    data = {'left': [-0.25, 0], 'bottom': [-1, 0.75], 'right': [0.25, 2], 'top': [1, 1.25]}
    callback.on_msg({'data': data})
    element = Rectangles([(-0.25, -1, 0.25, 1), (0, 0.75, 2, 1.25)])
    self.assertEqual(box_edit.element, element)
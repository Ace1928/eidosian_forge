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
def test_poly_draw_callback_with_vdims(self):
    polys = Polygons([{'x': [0, 2, 4], 'y': [0, 2, 0], 'A': 1}], vdims=['A'])
    poly_draw = PolyDraw(source=polys)
    plot = bokeh_server_renderer.get_plot(polys)
    self.assertIsInstance(plot.callbacks[0], PolyDrawCallback)
    callback = plot.callbacks[0]
    data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]], 'A': [1, 2]}
    callback.on_msg({'data': data})
    element = Polygons([{'x': [1, 2, 3], 'y': [1, 2, 3], 'A': 1}, {'x': [3, 4, 5], 'y': [3, 4, 5], 'A': 2}], vdims=['A'])
    self.assertEqual(poly_draw.element, element)
import datetime as dt
from unittest import SkipTest
import numpy as np
import panel as pn
import pytest
from bokeh.document import Document
from bokeh.models import (
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.util import dt_to_int
from holoviews.element import Curve, HeatMap, Image, Labels, Scatter
from holoviews.plotting.util import process_cmap
from holoviews.streams import PointDraw, Stream
from holoviews.util import render
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_explicit_categorical_cmap_on_integer_data(self):
    explicit_mapping = dict([(0, 'blue'), (1, 'red'), (2, 'green'), (3, 'purple')])
    points = Scatter(([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]), vdims=['y', 'Category']).opts(color_index='Category', cmap=explicit_mapping)
    plot = bokeh_renderer.get_plot(points)
    cmapper = plot.handles['color_mapper']
    cds = plot.handles['cds']
    self.assertEqual(cds.data['Category_str__'], ['0', '1', '2', '3'])
    self.assertEqual(cmapper.factors, ['0', '1', '2', '3'])
    self.assertEqual(cmapper.palette, ['blue', 'red', 'green', 'purple'])
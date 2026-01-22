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
def test_element_title_format(self):
    title_str = 'Label: {label}, group: {group}, dims: {dimensions}, type: {type}'
    e = Scatter([], label='the_label', group='the_group').opts(title=title_str)
    title = 'Label: the_label, group: the_group, dims: , type: Scatter'
    self.assertEqual(render(e).title.text, title)
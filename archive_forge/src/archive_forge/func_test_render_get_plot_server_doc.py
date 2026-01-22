from io import BytesIO
from unittest import SkipTest
import numpy as np
import panel as pn
import param
import pytest
from bokeh.io import curdoc
from bokeh.themes.theme import Theme
from panel.widgets import DiscreteSlider, FloatSlider, Player
from pyviz_comms import CommManager
from holoviews import Curve, DynamicMap, GridSpace, HoloMap, Image, Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer
from holoviews.plotting.bokeh import BokehRenderer
from holoviews.streams import Stream
def test_render_get_plot_server_doc(self):
    renderer = self.renderer.instance(mode='server')
    plot = renderer.get_plot(self.image1)
    self.assertIs(plot.document, curdoc())
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
def test_theme_rendering(self):
    attrs = {'figure': {'outline_line_color': '#444444'}}
    theme = Theme(json={'attrs': attrs})
    self.renderer.theme = theme
    plot = self.renderer.get_plot(Curve([]))
    self.renderer.components(plot, 'html')
    self.assertEqual(plot.state.outline_line_color, '#444444')
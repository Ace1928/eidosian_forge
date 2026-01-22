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
def test_get_size_grid_plot(self):
    grid = GridSpace({(i, j): self.image1 for i in range(3) for j in range(3)})
    plot = self.renderer.get_plot(grid)
    w, h = self.renderer.get_size(plot)
    self.assertEqual((w, h), (446, 437))
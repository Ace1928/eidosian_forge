import subprocess
from unittest import SkipTest
import numpy as np
import panel as pn
import param
from matplotlib import style
from panel.widgets import DiscreteSlider, FloatSlider, Player
from pyviz_comms import CommManager
from holoviews import Curve, DynamicMap, GridSpace, HoloMap, Image, ItemTable, Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.mpl import CurvePlot, MPLRenderer
from holoviews.plotting.renderer import Renderer
from holoviews.streams import Stream
def test_render_gif(self):
    data, metadata = self.renderer.components(self.map1, 'gif')
    self.assertIn("<img src='data:image/gif", data['text/html'])
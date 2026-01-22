import datetime as dt
import numpy as np
import pandas as pd
import pytest
from bokeh.models import FactorRange, FixedTicker
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.options import AbbreviatedException, Cycle, Palette
from holoviews.element import Curve
from holoviews.plotting.bokeh.callbacks import Callback, PointerXCallback
from holoviews.plotting.util import rgb2hex
from holoviews.streams import PointerX
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_cyclic_palette_curves(self):
    palette = Palette('Set1')
    hmap = HoloMap({i: NdOverlay({j: Curve(np.random.rand(3)).opts(color=palette) for j in range(3)}) for i in range(3)})
    colors = palette[3].values
    plot = bokeh_renderer.get_plot(hmap)
    for subp, color in zip(plot.subplots.values(), colors):
        color = color if isinstance(color, str) else rgb2hex(color)
        self.assertEqual(subp.handles['glyph'].line_color, color)
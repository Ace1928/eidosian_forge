import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_hover_ensure_vdims_sanitized(self):
    obj = Bars(np.random.rand(10, 2), vdims=['Dim with spaces'])
    obj = obj.opts(tools=['hover'])
    self._test_hover_info(obj, [('x', '@{x}'), ('Dim with spaces', '@{Dim_with_spaces}')])
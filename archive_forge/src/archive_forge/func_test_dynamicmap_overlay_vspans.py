import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_dynamicmap_overlay_vspans(self):
    el = hv.VSpans(data=[[1, 3], [2, 4]])
    dmap = hv.DynamicMap(lambda: hv.Overlay([el]))
    plot_el = bokeh_renderer.get_plot(el)
    plot_dmap = bokeh_renderer.get_plot(dmap)
    assert plot_el.handles['x_range'].start == plot_dmap.handles['x_range'].start
    assert plot_el.handles['x_range'].end == plot_dmap.handles['x_range'].end
    assert plot_el.handles['y_range'].start == plot_dmap.handles['y_range'].start
    assert plot_el.handles['y_range'].end == plot_dmap.handles['y_range'].end
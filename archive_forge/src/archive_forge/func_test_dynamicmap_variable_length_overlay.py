import numpy as np
import panel as pn
from bokeh.models import FactorRange, FixedTicker, HoverTool, Range1d, Span
from holoviews.core import DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.element import (
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream, Tap
from holoviews.util import Dynamic
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_dynamicmap_variable_length_overlay(self):
    selected = Stream.define('selected', items=[1])()

    def callback(items):
        return Overlay([Box(0, 0, radius * 2) for radius in items])
    dmap = DynamicMap(callback, streams=[selected])
    plot = bokeh_renderer.get_plot(dmap)
    assert len(plot.subplots) == 1
    selected.event(items=[1, 2, 4])
    assert len(plot.subplots) == 3
    selected.event(items=[1, 4])
    sp1, sp2, sp3 = plot.subplots.values()
    assert sp1.handles['cds'].data['xs'][0].min() == -1
    assert sp1.handles['glyph_renderer'].visible
    assert sp2.handles['cds'].data['xs'][0].min() == -4
    assert sp2.handles['glyph_renderer'].visible
    assert sp3.handles['cds'].data['xs'][0].min() == -4
    assert not sp3.handles['glyph_renderer'].visible
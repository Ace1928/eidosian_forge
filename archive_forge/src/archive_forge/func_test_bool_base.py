import numpy as np
import pytest
from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bool_base(self):
    overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
    plot = bokeh_renderer.get_plot(overlay)
    assert plot.subcoordinate_y is True
    assert len(plot.state.yaxis) == 1
    assert len(plot.subplots) == 2
    assert ('Curve', 'Data_0') in plot.subplots
    assert ('Curve', 'Data_1') in plot.subplots
    sp1 = plot.subplots['Curve', 'Data_0']
    assert sp1.handles['glyph_renderer'].coordinates.y_target.start == -0.5
    assert sp1.handles['glyph_renderer'].coordinates.y_target.end == 0.5
    sp2 = plot.subplots['Curve', 'Data_1']
    assert sp2.handles['glyph_renderer'].coordinates.y_target.start == 0.5
    assert sp2.handles['glyph_renderer'].coordinates.y_target.end == 1.5
    assert plot.handles['y_range'].start == -0.5
    assert plot.handles['y_range'].end == 1.5
    assert plot.handles['extra_y_ranges'] == {}
    assert plot.state.yaxis.ticker.ticks == [0, 1]
    assert plot.state.yaxis.major_label_overrides == {0: 'Data 0', 1: 'Data 1'}
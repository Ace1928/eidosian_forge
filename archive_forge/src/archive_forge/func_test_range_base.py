import numpy as np
import pytest
from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from .test_plot import TestBokehPlot, bokeh_renderer
def test_range_base(self):
    overlay = Overlay([Curve(range(10), label='Data 0').opts(subcoordinate_y=(0, 0.5)), Curve(range(10), label='Data 1').opts(subcoordinate_y=(0.5, 1))])
    plot = bokeh_renderer.get_plot(overlay)
    assert plot.subcoordinate_y == (0, 0.5)
    assert len(plot.state.yaxis) == 1
    assert len(plot.subplots) == 2
    assert ('Curve', 'Data_0') in plot.subplots
    assert ('Curve', 'Data_1') in plot.subplots
    sp1 = plot.subplots['Curve', 'Data_0']
    assert sp1.handles['glyph_renderer'].coordinates.y_target.start == 0
    assert sp1.handles['glyph_renderer'].coordinates.y_target.end == 0.5
    sp2 = plot.subplots['Curve', 'Data_1']
    assert sp2.handles['glyph_renderer'].coordinates.y_target.start == 0.5
    assert sp2.handles['glyph_renderer'].coordinates.y_target.end == 1
    assert plot.handles['y_range'].start == 0
    assert plot.handles['y_range'].end == 1
    assert plot.handles['extra_y_ranges'] == {}
    assert plot.state.yaxis.ticker.ticks == [0.25, 0.75]
    assert plot.state.yaxis.major_label_overrides == {0.25: 'Data 0', 0.75: 'Data 1'}
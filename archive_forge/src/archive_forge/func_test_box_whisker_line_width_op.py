import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, ColumnDataSource, LinearColorMapper
from holoviews.element import BoxWhisker
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_box_whisker_line_width_op(self):
    a = np.repeat(np.arange(5), 5)
    b = np.repeat(np.arange(5), 5)
    box = BoxWhisker((a, b, np.arange(25)), ['a', 'b'], 'd').opts(box_line_width='b')
    plot = bokeh_renderer.get_plot(box)
    source = plot.handles['vbar_1_source']
    glyph = plot.handles['vbar_1_glyph']
    self.assertEqual(source.data['box_line_width'], np.arange(5))
    self.assertEqual(property_to_dict(glyph.line_width), {'field': 'box_line_width'})
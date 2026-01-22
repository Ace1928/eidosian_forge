import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_labels_color_mapped_text_vals(self):
    labels = Labels([(0, 1, 0.33333), (1, 0, 0.66666)]).opts(color_index=2)
    plot = bokeh_renderer.get_plot(labels)
    source = plot.handles['source']
    glyph = plot.handles['glyph']
    cmapper = plot.handles['color_mapper']
    expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]), 'Label': ['0.33333', '0.66666'], 'text_color': np.array([0.33333, 0.66666])}
    for k, vals in expected.items():
        self.assertEqual(source.data[k], vals)
    self.assertEqual(glyph.x, 'x')
    self.assertEqual(glyph.y, 'y')
    self.assertEqual(glyph.text, 'Label')
    self.assertEqual(property_to_dict(glyph.text_color), {'field': 'text_color', 'transform': cmapper})
    self.assertEqual(cmapper.low, 0.33333)
    self.assertEqual(cmapper.high, 0.66666)
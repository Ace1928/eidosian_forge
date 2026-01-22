import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_labels_formatter(self):
    vdim = Dimension('text', value_format=lambda x: f'{x:.1f}')
    labels = Labels([(0, 1, 0.33333), (1, 0, 0.66666)], vdims=vdim)
    plot = bokeh_renderer.get_plot(labels)
    source = plot.handles['source']
    glyph = plot.handles['glyph']
    expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]), 'text': ['0.3', '0.7']}
    for k, vals in expected.items():
        self.assertEqual(source.data[k], vals)
    self.assertEqual(glyph.x, 'x')
    self.assertEqual(glyph.y, 'y')
    self.assertEqual(glyph.text, 'text')
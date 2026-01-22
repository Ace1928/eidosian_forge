from unittest import SkipTest
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.element import Violin
from holoviews.operation.stats import univariate_kde
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_violin_split_op_multi(self):
    a = np.repeat(np.arange(5), 5)
    b = np.repeat(np.arange(5), 5)
    violin = Violin((a, b, np.arange(25)), ['a', 'b'], 'd').opts(split=dim('b') > 2)
    plot = bokeh_renderer.get_plot(violin)
    source = plot.handles['patches_1_source']
    glyph = plot.handles['patches_1_glyph']
    cmapper = plot.handles['violin_color_mapper']
    values = ['False', 'True', 'False', 'True', 'False', 'True', 'False', 'True', 'False', 'True']
    self.assertEqual(source.data["dim('b')>2"], values)
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': "dim('b')>2", 'transform': cmapper})
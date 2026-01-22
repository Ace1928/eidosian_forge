import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, FactorRange, LinearColorMapper, Scatter
from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Points
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_point_categorical_dtype_color_op(self):
    df = pd.DataFrame(dict(sample_id=['subject 1', 'subject 2', 'subject 3', 'subject 4'], category=['apple', 'pear', 'apple', 'pear'], value=[1, 2, 3, 4]))
    df['category'] = df['category'].astype('category')
    points = Points(df, ['sample_id', 'value']).opts(color='category')
    plot = bokeh_renderer.get_plot(points)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    cmapper = plot.handles['color_color_mapper']
    self.assertTrue(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['apple', 'pear'])
    self.assertEqual(np.asarray(cds.data['color']), np.array(['apple', 'pear', 'apple', 'pear']))
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})
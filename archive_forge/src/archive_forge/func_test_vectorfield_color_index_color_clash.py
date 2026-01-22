import numpy as np
from holoviews.element import VectorField
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vectorfield_color_index_color_clash(self):
    vectorfield = VectorField([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims='color').opts(line_color='color', color_index='color')
    with ParamLogStream() as log:
        bokeh_renderer.get_plot(vectorfield)
    log_msg = log.stream.read()
    warning = "The `color_index` parameter is deprecated in favor of color style mapping, e.g. `color=dim('color')` or `line_color=dim('color')`\nCannot declare style mapping for 'line_color' option and declare a color_index; ignoring the color_index.\n"
    self.assertEqual(log_msg, warning)
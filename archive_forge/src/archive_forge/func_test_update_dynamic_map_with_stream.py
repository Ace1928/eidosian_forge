from unittest.mock import Mock
import numpy as np
import panel as pn
from bokeh.document import Document
from pyviz_comms import Comm
import holoviews as hv
from holoviews.streams import (
from .test_plot import TestPlotlyPlot
def test_update_dynamic_map_with_stream(self):
    ys = np.arange(10)
    Scale = Stream.define('Scale', scale=1.0)
    scale_stream = Scale()

    def build_scatter(scale):
        return hv.Scatter(ys * scale)
    dmap = hv.DynamicMap(build_scatter, streams=[scale_stream])
    dmap_pane = pn.pane.HoloViews(dmap, backend='plotly')
    doc = Document()
    comm = Comm()
    dmap_pane.get_root(doc, comm)
    _, plotly_pane = next(iter(dmap_pane._plots.values()))
    data = plotly_pane.object['data']
    self.assertEqual(len(data), 1)
    self.assertEqual(data[0]['type'], 'scatter')
    np.testing.assert_equal(data[0]['y'], ys)
    fn = Mock()
    plotly_pane.param.watch(fn, 'object')
    scale_stream.event(scale=2.0)
    data = plotly_pane.object['data']
    np.testing.assert_equal(data[0]['y'], ys * 2.0)
    fn.assert_called_once()
    args, kwargs = fn.call_args_list[0]
    event = args[0]
    self.assertIs(event.obj, plotly_pane)
    self.assertIs(event.new, plotly_pane.object)
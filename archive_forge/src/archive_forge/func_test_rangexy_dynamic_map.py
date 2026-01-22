import unittest
from unittest.mock import MagicMock, patch
from dash._callback_context import CallbackContext
from holoviews import Bounds, DynamicMap, Scatter
from holoviews.plotting.plotly.dash import (
from holoviews.streams import BoundsXY, RangeXY, Selection1D
from .test_plot import TestPlotlyPlot
import plotly.io as pio
def test_rangexy_dynamic_map(self):
    scatter = Scatter([[0, 1], [0, 1]], kdims=['x'], vdims=['y'])
    rangexy = RangeXY(source=scatter)

    def dmap_fn(x_range, y_range):
        x_range = (0, 1) if x_range is None else x_range
        y_range = (0, 1) if y_range is None else y_range
        return Scatter([[x_range[0], y_range[0]], [x_range[1], y_range[1]]], kdims=['x1'], vdims=['y1'])
    dmap = DynamicMap(dmap_fn, streams=[rangexy])
    components = to_dash(self.app, [scatter, dmap], reset_button=True)
    self.assertIsInstance(components, DashComponents)
    self.assertEqual(len(components.graphs), 2)
    self.assertEqual(len(components.kdims), 0)
    self.assertIsInstance(components.store, Store)
    self.assertEqual(len(components.resets), 1)
    decorator_args = next(iter(self.app.callback.call_args_list[0]))
    outputs, inputs, states = decorator_args
    expected_outputs = [(g.id, 'figure') for g in components.graphs] + [(components.store.id, 'data')]
    self.assertEqual([(output.component_id, output.component_property) for output in outputs], expected_outputs)
    expected_inputs = [(g.id, prop) for g in components.graphs for prop in ['selectedData', 'relayoutData']] + [(components.resets[0].id, 'n_clicks')]
    self.assertEqual([(ip.component_id, ip.component_property) for ip in inputs], expected_inputs)
    expected_state = [(components.store.id, 'data')]
    self.assertEqual([(state.component_id, state.component_property) for state in states], expected_state)
    callback_fn = self.app.callback.return_value.call_args[0][0]
    store_value = encode_store_data({'streams': {id(rangexy): rangexy.contents}})
    with patch.object(CallbackContext, 'triggered', [{'prop_id': components.graphs[0].id + '.relayoutData'}]):
        [fig1, fig2, new_store] = callback_fn({}, {'xaxis.range[0]': 1, 'xaxis.range[1]': 3, 'yaxis.range[0]': 2, 'yaxis.range[1]': 4}, {}, {}, None, store_value)
    self.assertEqual(fig1['data'][0]['type'], 'scatter')
    self.assertEqual(len(fig2['data']), 1)
    self.assertEqual(list(fig2['data'][0]['x']), [1, 3])
    self.assertEqual(list(fig2['data'][0]['y']), [2, 4])
    self.assertEqual(decode_store_data(new_store), {'streams': {id(rangexy): {'x_range': (1, 3), 'y_range': (2, 4)}}, 'kdims': {}})
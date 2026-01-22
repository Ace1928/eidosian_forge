import unittest
from unittest.mock import MagicMock, patch
from dash._callback_context import CallbackContext
from holoviews import Bounds, DynamicMap, Scatter
from holoviews.plotting.plotly.dash import (
from holoviews.streams import BoundsXY, RangeXY, Selection1D
from .test_plot import TestPlotlyPlot
import plotly.io as pio
def test_kdims_dynamic_map(self):
    dmap = DynamicMap(lambda kdim1: Scatter([kdim1, kdim1]), kdims=['kdim1']).redim.values(kdim1=[1, 2, 3, 4])
    components = to_dash(self.app, [dmap])
    self.assertIsInstance(components, DashComponents)
    self.assertEqual(len(components.graphs), 1)
    self.assertEqual(len(components.kdims), 1)
    self.assertIsInstance(components.store, Store)
    self.assertEqual(len(components.resets), 0)
    decorator_args = next(iter(self.app.callback.call_args_list[0]))
    outputs, inputs, states = decorator_args
    expected_outputs = [(g.id, 'figure') for g in components.graphs] + [(components.store.id, 'data')]
    self.assertEqual([(output.component_id, output.component_property) for output in outputs], expected_outputs)
    expected_inputs = [(g.id, prop) for g in components.graphs for prop in ['selectedData', 'relayoutData']] + [(next(iter(components.kdims.values())).children[1].id, 'value')]
    self.assertEqual([(ip.component_id, ip.component_property) for ip in inputs], expected_inputs)
    expected_state = [(components.store.id, 'data')]
    self.assertEqual([(state.component_id, state.component_property) for state in states], expected_state)
    callback_fn = self.decorator.call_args_list[0][0][0]
    store_value = encode_store_data({'streams': {}})
    with patch.object(CallbackContext, 'triggered', []):
        [fig, new_store] = callback_fn({}, {}, 3, None, store_value)
    self.assertEqual(fig['data'][0]['type'], 'scatter')
    self.assertEqual(list(fig['data'][0]['x']), [0, 1])
    self.assertEqual(list(fig['data'][0]['y']), [3, 3])
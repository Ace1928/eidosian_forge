import unittest
from unittest.mock import MagicMock, patch
from dash._callback_context import CallbackContext
from holoviews import Bounds, DynamicMap, Scatter
from holoviews.plotting.plotly.dash import (
from holoviews.streams import BoundsXY, RangeXY, Selection1D
from .test_plot import TestPlotlyPlot
import plotly.io as pio
def test_boundsxy_dynamic_map(self):
    scatter = Scatter([0, 0])
    boundsxy = BoundsXY(source=scatter)
    dmap = DynamicMap(lambda bounds: Bounds(bounds) if bounds is not None else Bounds((0, 0, 0, 0)), streams=[boundsxy])
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
    fig1 = components.graphs[0].figure
    fig2 = components.graphs[1].figure
    self.assertEqual(fig1['data'][0]['type'], 'scatter')
    self.assertEqual(len(fig2['data']), 0)
    self.assertEqual(len(fig2['layout']['shapes']), 1)
    self.assertEqual(fig2['layout']['shapes'][0]['path'], 'M0 0L0 0L0 0L0 0L0 0Z')
    callback_fn = self.app.callback.return_value.call_args[0][0]
    store_value = encode_store_data({'streams': {id(boundsxy): boundsxy.contents}})
    with patch.object(CallbackContext, 'triggered', [{'prop_id': inputs[0].component_id + '.selectedData'}]):
        [fig1, fig2, new_store] = callback_fn({'range': {'x': [1, 2], 'y': [3, 4]}}, {}, {}, {}, 0, store_value)
    self.assertEqual(fig1['data'][0]['type'], 'scatter')
    self.assertEqual(len(fig2['data']), 0)
    self.assertEqual(len(fig2['layout']['shapes']), 1)
    self.assertEqual(fig2['layout']['shapes'][0]['path'], 'M1 3L1 4L2 4L2 3L1 3Z')
    self.assertEqual(decode_store_data(new_store), {'streams': {id(boundsxy): {'bounds': (1, 3, 2, 4)}}, 'kdims': {}})
    with patch.object(CallbackContext, 'triggered', [{'prop_id': components.resets[0].id + '.n_clicks'}]):
        [fig1, fig2, new_store] = callback_fn({'range': {'x': [1, 2], 'y': [3, 4]}}, {}, {}, {}, 1, store_value)
    self.assertEqual(fig1['data'][0]['type'], 'scatter')
    self.assertEqual(len(fig2['data']), 0)
    self.assertEqual(len(fig2['layout']['shapes']), 1)
    self.assertEqual(fig2['layout']['shapes'][0]['path'], 'M0 0L0 0L0 0L0 0L0 0Z')
    self.assertEqual(decode_store_data(new_store), {'streams': {id(boundsxy): {'bounds': None}}, 'reset_nclicks': 1, 'kdims': {}})
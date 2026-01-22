import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testSelection1DCallback(self):
    plots, streamss, callbacks, sel_events = build_callback_set(Selection1DCallback, ['first', 'second', 'third', 'forth', 'other'], Selection1D, 2)
    selected_data1 = {'points': [{'pointNumber': 0, 'curveNumber': 0}, {'pointNumber': 2, 'curveNumber': 0}]}
    Selection1DCallback.update_streams_from_property_update('selected_data', selected_data1, self.fig_dict)
    for stream, events in zip(streamss[0], sel_events[0]):
        assert stream.index == [0, 2]
        assert len(events) == 1
    for stream in streamss[1] + streamss[2] + streamss[3]:
        assert stream.index == []
    selected_data1 = {'points': [{'pointNumber': 0, 'curveNumber': 0}, {'pointNumber': 1, 'curveNumber': 0}, {'pointNumber': 1, 'curveNumber': 1}, {'pointNumber': 2, 'curveNumber': 1}]}
    Selection1DCallback.update_streams_from_property_update('selected_data', selected_data1, self.fig_dict)
    for stream in streamss[0]:
        assert stream.index == [0, 1]
    for stream in streamss[1]:
        assert stream.index == [1, 2]
    for stream in streamss[2] + streamss[3]:
        assert stream.index == []
    selected_data1 = {'points': [{'pointNumber': 0, 'curveNumber': 3}, {'pointNumber': 2, 'curveNumber': 3}]}
    Selection1DCallback.update_streams_from_property_update('selected_data', selected_data1, self.fig_dict)
    for stream, _events in zip(streamss[3], sel_events[3]):
        assert stream.index == [0, 2]
    for _stream, events in zip(streamss[4], sel_events[4]):
        assert len(events) == 0
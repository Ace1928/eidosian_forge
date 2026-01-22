import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testSelection1DCallbackEventData(self):
    selected_data1 = {'points': [{'pointNumber': 0, 'curveNumber': 0}, {'pointNumber': 2, 'curveNumber': 0}]}
    event_data = Selection1DCallback.get_event_data_from_property_update('selected_data', selected_data1, self.fig_dict)
    self.assertEqual(event_data, {'first': {'index': [0, 2]}, 'second': {'index': []}, 'third': {'index': []}, 'forth': {'index': []}})
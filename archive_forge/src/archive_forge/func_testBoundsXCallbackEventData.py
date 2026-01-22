import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testBoundsXCallbackEventData(self):
    selected_data1 = {'range': {'x': [1, 4], 'y': [-1, 5]}}
    event_data = BoundsXCallback.get_event_data_from_property_update('selected_data', selected_data1, self.fig_dict)
    self.assertEqual(event_data, {'first': {'boundsx': (1, 4)}, 'second': {'boundsx': (1, 4)}, 'third': {'boundsx': None}, 'forth': {'boundsx': None}})
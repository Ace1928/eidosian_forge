import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testBoundsXYCallbackEventData(self):
    selected_data1 = {'range': {'x': [1, 4], 'y': [-1, 5]}}
    event_data = BoundsXYCallback.get_event_data_from_property_update('selected_data', selected_data1, self.fig_dict)
    self.assertEqual(event_data, {'first': {'bounds': (1, -1, 4, 5)}, 'second': {'bounds': (1, -1, 4, 5)}, 'third': {'bounds': None}, 'forth': {'bounds': None}})
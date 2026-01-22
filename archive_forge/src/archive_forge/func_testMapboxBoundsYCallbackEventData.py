import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testMapboxBoundsYCallbackEventData(self):
    selected_data = {'range': {'mapbox3': [[self.lon_range1[0], self.lat_range1[0]], [self.lon_range1[1], self.lat_range1[1]]]}}
    event_data = BoundsYCallback.get_event_data_from_property_update('selected_data', selected_data, self.mapbox_fig_dict)
    self.assertEqual(event_data, {'first': {'boundsy': None}, 'second': {'boundsy': None}, 'third': {'boundsy': (self.northing_range1[0], self.northing_range1[1])}})
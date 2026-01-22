from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
@classmethod
def build_event_data_from_relayout_data(cls, traces, property_value):
    event_data = {}
    for trace in traces:
        trace_type = trace.get('type', 'scattermapbox')
        trace_uid = trace.get('uid', None)
        if _trace_to_subplot.get(trace_type, None) != ['mapbox']:
            continue
        subplot_id = trace.get('subplot', 'mapbox')
        derived_prop = subplot_id + '._derived'
        if not property_value:
            x_range = None
            y_range = None
        elif 'coordinates' in property_value.get(derived_prop, {}):
            coords = property_value[derived_prop]['coordinates']
            (lon_top_left, lat_top_left), (lon_top_right, lat_top_right), (lon_bottom_right, lat_bottom_right), (lon_bottom_left, lat_bottom_left) = coords
            lon_left = min(lon_top_left, lon_bottom_left)
            lon_right = max(lon_top_right, lon_bottom_right)
            lat_bottom = min(lat_bottom_left, lat_bottom_right)
            lat_top = max(lat_top_left, lat_top_right)
            x_range, y_range = Tiles.lon_lat_to_easting_northing([lon_left, lon_right], [lat_bottom, lat_top])
            x_range = tuple(x_range)
            y_range = tuple(y_range)
        else:
            continue
        stream_data = {}
        if cls.x_range:
            stream_data['x_range'] = x_range
        if cls.y_range:
            stream_data['y_range'] = y_range
        event_data[trace_uid] = stream_data
    return event_data
from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
@classmethod
def update_event_data_mapbox(cls, range_data, traces, event_data):
    for trace in traces:
        trace_type = trace.get('type', 'scatter')
        trace_uid = trace.get('uid', None)
        if _trace_to_subplot.get(trace_type, None) != ['mapbox']:
            continue
        mapbox_ref = trace.get('subplot', 'mapbox')
        if mapbox_ref in range_data:
            lon_bounds = [range_data[mapbox_ref][0][0], range_data[mapbox_ref][1][0]]
            lat_bounds = [range_data[mapbox_ref][0][1], range_data[mapbox_ref][1][1]]
            easting, northing = Tiles.lon_lat_to_easting_northing(lon_bounds, lat_bounds)
            new_bounds = (easting[0], northing[0], easting[1], northing[1])
            if cls.boundsx and cls.boundsy:
                stream_data = dict(bounds=new_bounds)
            elif cls.boundsx:
                stream_data = dict(boundsx=(new_bounds[0], new_bounds[2]))
            elif cls.boundsy:
                stream_data = dict(boundsy=(new_bounds[1], new_bounds[3]))
            else:
                stream_data = {}
            event_data[trace_uid] = stream_data
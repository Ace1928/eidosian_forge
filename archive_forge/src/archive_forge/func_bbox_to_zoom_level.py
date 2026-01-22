import math
from ..bindings.view_state import ViewState
from .type_checking import is_pandas_df
def bbox_to_zoom_level(bbox):
    """Computes the zoom level of a lat/lng bounding box

    Parameters
    ----------
    bbox : list of list of float
        Northwest and southeast corners of a bounding box, given as two points in a list

    Returns
    -------
    int
        Zoom level of map in a WGS84 Mercator projection (e.g., like that of Google Maps)
    """
    lat_diff = max(bbox[0][0], bbox[1][0]) - min(bbox[0][0], bbox[1][0])
    lng_diff = max(bbox[0][1], bbox[1][1]) - min(bbox[0][1], bbox[1][1])
    max_diff = max(lng_diff, lat_diff)
    zoom_level = None
    if max_diff < 360.0 / math.pow(2, 20):
        zoom_level = 21
    else:
        zoom_level = int(-1 * (math.log(max_diff) / math.log(2.0) - math.log(360.0) / math.log(2)))
        if zoom_level < 1:
            zoom_level = 1
    return zoom_level
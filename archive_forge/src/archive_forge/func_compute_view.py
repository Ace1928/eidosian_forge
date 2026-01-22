import math
from ..bindings.view_state import ViewState
from .type_checking import is_pandas_df
def compute_view(points, view_proportion=1, view_type=ViewState):
    """Automatically computes a zoom level for the points passed in.

    Parameters
    ----------
    points : list of list of float or pandas.DataFrame
        A list of points
    view_propotion : float, default 1
        Proportion of the data that is meaningful to plot
    view_type : class constructor for pydeck.ViewState, default :class:`pydeck.bindings.view_state.ViewState`
        Class constructor for a viewport. In the current version of pydeck,
        users most likely do not have to modify this attribute.

    Returns
    -------
    pydeck.Viewport
        Viewport fitted to the data
    """
    if is_pandas_df(points):
        points = points.to_records(index=False)
    bbox = get_bbox(get_n_pct(points, view_proportion))
    zoom = bbox_to_zoom_level(bbox)
    center = geometric_mean(points)
    instance = view_type(latitude=center[1], longitude=center[0], zoom=zoom)
    return instance
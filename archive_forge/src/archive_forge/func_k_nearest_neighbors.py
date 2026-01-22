import math
from ..bindings.view_state import ViewState
from .type_checking import is_pandas_df
def k_nearest_neighbors(points, center, k):
    """Gets the k furthest points from the center

    Parameters
    ----------
    points : list of list of float
        List of (x, y) coordinates
    center : list of list of float
        Center point
    k : int
        Number of points

    Returns
    -------
    list
        Index of the k furthest points

    Todo
    ---
    Currently implemently naively, needs to be more efficient
    """
    pts_with_distance = [(pt, euclidean(pt, center)) for pt in points]
    sorted_pts = sorted(pts_with_distance, key=lambda x: x[1])
    return [x[0] for x in sorted_pts][:int(k)]
from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
def find_bezier_t_intersecting_with_closedpath(bezier_point_at_t, inside_closedpath, t0=0.0, t1=1.0, tolerance=0.01):
    """
    Find the intersection of the Bézier curve with a closed path.

    The intersection point *t* is approximated by two parameters *t0*, *t1*
    such that *t0* <= *t* <= *t1*.

    Search starts from *t0* and *t1* and uses a simple bisecting algorithm
    therefore one of the end points must be inside the path while the other
    doesn't. The search stops when the distance of the points parametrized by
    *t0* and *t1* gets smaller than the given *tolerance*.

    Parameters
    ----------
    bezier_point_at_t : callable
        A function returning x, y coordinates of the Bézier at parameter *t*.
        It must have the signature::

            bezier_point_at_t(t: float) -> tuple[float, float]

    inside_closedpath : callable
        A function returning True if a given point (x, y) is inside the
        closed path. It must have the signature::

            inside_closedpath(point: tuple[float, float]) -> bool

    t0, t1 : float
        Start parameters for the search.

    tolerance : float
        Maximal allowed distance between the final points.

    Returns
    -------
    t0, t1 : float
        The Bézier path parameters.
    """
    start = bezier_point_at_t(t0)
    end = bezier_point_at_t(t1)
    start_inside = inside_closedpath(start)
    end_inside = inside_closedpath(end)
    if start_inside == end_inside and start != end:
        raise NonIntersectingPathException('Both points are on the same side of the closed path')
    while True:
        if np.hypot(start[0] - end[0], start[1] - end[1]) < tolerance:
            return (t0, t1)
        middle_t = 0.5 * (t0 + t1)
        middle = bezier_point_at_t(middle_t)
        middle_inside = inside_closedpath(middle)
        if start_inside ^ middle_inside:
            t1 = middle_t
            if end == middle:
                return (t0, t1)
            end = middle
        else:
            t0 = middle_t
            if start == middle:
                return (t0, t1)
            start = middle
            start_inside = middle_inside
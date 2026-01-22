import numpy as np
from scipy.spatial import cKDTree
from ._hough_transform import _hough_circle, _hough_ellipse, _hough_line
from ._hough_transform import _probabilistic_hough_line as _prob_hough_line
def label_distant_points(xs, ys, min_xdistance, min_ydistance, max_points):
    """Keep points that are separated by certain distance in each dimension.

    The first point is always accepted and all subsequent points are selected
    so that they are distant from all their preceding ones.

    Parameters
    ----------
    xs : array, shape (M,)
        X coordinates of points.
    ys : array, shape (M,)
        Y coordinates of points.
    min_xdistance : int
        Minimum distance separating points in the x dimension.
    min_ydistance : int
        Minimum distance separating points in the y dimension.
    max_points : int
        Max number of distant points to keep.

    Returns
    -------
    should_keep : array of bool
        A mask array for distant points to keep.
    """
    is_neighbor = np.zeros(len(xs), dtype=bool)
    coordinates = np.stack([xs, ys], axis=1)
    kd_tree = cKDTree(coordinates)
    n_pts = 0
    for i in range(len(xs)):
        if n_pts >= max_points:
            is_neighbor[i] = True
        elif not is_neighbor[i]:
            neighbors_i = kd_tree.query_ball_point((xs[i], ys[i]), np.hypot(min_xdistance, min_ydistance))
            for ni in neighbors_i:
                x_close = abs(xs[ni] - xs[i]) <= min_xdistance
                y_close = abs(ys[ni] - ys[i]) <= min_ydistance
                if x_close and y_close and (ni > i):
                    is_neighbor[ni] = True
            n_pts += 1
    should_keep = ~is_neighbor
    return should_keep
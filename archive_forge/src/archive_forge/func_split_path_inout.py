from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
def split_path_inout(path, inside, tolerance=0.01, reorder_inout=False):
    """
    Divide a path into two segments at the point where ``inside(x, y)`` becomes
    False.
    """
    from .path import Path
    path_iter = path.iter_segments()
    ctl_points, command = next(path_iter)
    begin_inside = inside(ctl_points[-2:])
    ctl_points_old = ctl_points
    iold = 0
    i = 1
    for ctl_points, command in path_iter:
        iold = i
        i += len(ctl_points) // 2
        if inside(ctl_points[-2:]) != begin_inside:
            bezier_path = np.concatenate([ctl_points_old[-2:], ctl_points])
            break
        ctl_points_old = ctl_points
    else:
        raise ValueError('The path does not intersect with the patch')
    bp = bezier_path.reshape((-1, 2))
    left, right = split_bezier_intersecting_with_closedpath(bp, inside, tolerance)
    if len(left) == 2:
        codes_left = [Path.LINETO]
        codes_right = [Path.MOVETO, Path.LINETO]
    elif len(left) == 3:
        codes_left = [Path.CURVE3, Path.CURVE3]
        codes_right = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    elif len(left) == 4:
        codes_left = [Path.CURVE4, Path.CURVE4, Path.CURVE4]
        codes_right = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        raise AssertionError('This should never be reached')
    verts_left = left[1:]
    verts_right = right[:]
    if path.codes is None:
        path_in = Path(np.concatenate([path.vertices[:i], verts_left]))
        path_out = Path(np.concatenate([verts_right, path.vertices[i:]]))
    else:
        path_in = Path(np.concatenate([path.vertices[:iold], verts_left]), np.concatenate([path.codes[:iold], codes_left]))
        path_out = Path(np.concatenate([verts_right, path.vertices[i:]]), np.concatenate([codes_right, path.codes[i:]]))
    if reorder_inout and (not begin_inside):
        path_in, path_out = (path_out, path_in)
    return (path_in, path_out)
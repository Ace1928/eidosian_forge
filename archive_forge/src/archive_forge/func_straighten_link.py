import spherogram
import random
import itertools
from . import pl_utils
from .rational_linear_algebra import QQ, Matrix, Vector3
from .exceptions import GeneralPositionError
def straighten_link(points_list):
    for i in range(len(points_list)):
        other_points = points_list.copy()
        del other_points[i]
        other_arcs = arcs_from_points(other_points)
        while True:
            points = points_list[i]
            tri = straightenable_tri(points, other_arcs)
            if tri:
                a, b, c = tri
                if a < b < c:
                    points = points[c:] + points[:b]
                else:
                    points = points[:b] + points[b + 1:]
            else:
                break
            points_list[i] = points
    return points_list
import spherogram
import random
import itertools
from . import pl_utils
from .rational_linear_algebra import QQ, Matrix, Vector3
from .exceptions import GeneralPositionError
def straighten_knot(points):
    while True:
        tri = straightenable_tri(points)
        if tri:
            a, b, c = tri
            if a < b < c:
                points = points[c:] + points[:b]
            else:
                points = points[:b] + points[b + 1:]
        else:
            break
    return points
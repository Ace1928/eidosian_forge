from ..snap.t3mlite import simplex
from ..hyperboloid import *
def check_points_equal(v0, v1):
    RF = v0[0].parent()
    if abs(r13_dot(v0, v0)) < RF(1e-10):
        if abs(r13_dot(v1, v1)) > RF(1e-10):
            raise Exception('Light-like vs time-like:', v0, v1)
        if abs(r13_dot(v0, v1)) > RF(1e-10):
            raise Exception('Non-colinlinear light like:', v0, v1)
    elif any((abs(x - y) > RF(1e-10) for x, y in zip(v0, v1))):
        raise Exception('Different time-like:', v0, v1)
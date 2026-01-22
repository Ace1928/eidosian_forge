from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def height_euclidean_triangle(z0, z1, z2):
    """
    Takes three (ideal) points in C subset C union { Infinity}
    regarded as boundary of the upper half space model. Returns
    the Euclidean height of the triangle spanned by the points.
    """
    return abs(_dist_from_projection(z0 - z1, z2 - z1))
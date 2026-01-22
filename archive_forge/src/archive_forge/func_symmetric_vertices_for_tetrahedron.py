from snappy.SnapPy import matrix
from ..upper_halfspace.ideal_point import Infinity
def symmetric_vertices_for_tetrahedron(z):
    """
    Given a tetrahedron shape, returns four (ideal) points spanning
    a tetrahedron of that shape.

    The points are in C subset C union { Infinity } regarded as
    boundary of the upper half space model.

    Duplicates initial_tetrahedron(... centroid_at_origin = true)
    in choose_generators.c.
    """
    w = z.sqrt() + (z - 1).sqrt()
    return [w, 1 / w, -1 / w, -w]
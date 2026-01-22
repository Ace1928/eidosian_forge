import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def join_strands(x, y):
    """
    Input: two (c, i) pairs where c is a Crossing, Strand, or Tangle object and i is an index into
    c.adjacent. Joins the objects by having them refer to each other at those positions.

    When c is a Tangle it is conceptually a special case since its c.adjacent is being
    used to record the boundary strands.

    This function equivalent to creating a Strand s with s.adjacent = [x, y] and then
    doing s.fuse()
    """
    (a, i), (b, j) = (x, y)
    a.adjacent[i] = (b, j)
    b.adjacent[j] = (a, i)
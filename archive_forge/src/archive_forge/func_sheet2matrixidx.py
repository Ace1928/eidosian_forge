import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def sheet2matrixidx(self, x, y):
    """
        Convert a point (x,y) in sheet coordinates to the integer row
        and column index of the matrix cell in which that point falls,
        given a bounds and density.  Returns (row,column).

        Note that if coordinates along the right or bottom boundary
        are passed into this function, the returned matrix coordinate
        of the boundary will be just outside the matrix, because the
        right and bottom boundaries are exclusive.

        Valid for scalar or array x and y.
        """
    r, c = self.sheet2matrix(x, y)
    r = np.floor(r)
    c = np.floor(c)
    if hasattr(r, 'astype'):
        return (r.astype(int), c.astype(int))
    else:
        return (int(r), int(c))
from collections import namedtuple
import math
import warnings
def to_shapely(self):
    """Return an affine transformation matrix compatible with shapely

        Shapely's affinity module expects an affine transformation matrix
        in (a,b,d,e,xoff,yoff) order.

        :rtype: tuple
        """
    return (self.a, self.b, self.d, self.e, self.xoff, self.yoff)
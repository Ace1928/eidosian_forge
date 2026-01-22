import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
@property
def minimum_rotated_rectangle(self):
    """
        Returns the oriented envelope (minimum rotated rectangle) that
        encloses the geometry.

        Unlike `envelope` this rectangle is not constrained to be parallel to the
        coordinate axes. If the convex hull of the object is a degenerate (line
        or point) this degenerate is returned.

        Alias of `oriented_envelope`.
        """
    return shapely.oriented_envelope(self)
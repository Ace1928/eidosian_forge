import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
def representative_point(self):
    """Returns a point guaranteed to be within the object, cheaply.

        Alias of `point_on_surface`.
        """
    return shapely.point_on_surface(self)
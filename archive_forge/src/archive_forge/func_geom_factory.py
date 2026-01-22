import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
def geom_factory(g, parent=None):
    """
    Creates a Shapely geometry instance from a pointer to a GEOS geometry.

    .. warning::
        The GEOS library used to create the the GEOS geometry pointer
        and the GEOS library used by Shapely must be exactly the same, or
        unexpected results or segfaults may occur.

    .. deprecated:: 2.0
        Deprecated in Shapely 2.0, and will be removed in a future version.
    """
    warn("The 'geom_factory' function is deprecated in Shapely 2.0, and will be removed in a future version", DeprecationWarning, stacklevel=2)
    return _geom_factory(g)
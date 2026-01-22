import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
def geometryType(self):
    warn("The 'GeometryType()' method is deprecated, and will be removed in the future. You can use the 'geom_type' attribute instead.", ShapelyDeprecationWarning, stacklevel=2)
    return self.geom_type
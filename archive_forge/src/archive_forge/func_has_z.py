import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
@property
def has_z(self):
    """True if the geometry's coordinate sequence(s) have z values (are
        3-dimensional)"""
    return bool(shapely.has_z(self))
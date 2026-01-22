import numpy as np
import shapely
from shapely.algorithms.cga import is_ccw_impl, signed_area
from shapely.errors import TopologicalError
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
@property
def interiors(self):
    if self.is_empty:
        return []
    return InteriorRingSequence(self)
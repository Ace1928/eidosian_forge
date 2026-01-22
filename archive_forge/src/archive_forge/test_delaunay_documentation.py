import unittest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import triangulate

    Only testing the number of triangles and their type here.
    This doesn't actually test the points in the resulting geometries.

    
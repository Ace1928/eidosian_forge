import unittest
import pytest
from shapely.geometry import LinearRing, LineString
from shapely.testing import assert_geometries_equal
def test_parallel_offset_linear_ring(self):
    lr1 = LinearRing([(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)])
    assert_geometries_equal(lr1.parallel_offset(2, 'left', resolution=1), LineString([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]))
    assert_geometries_equal(lr1.offset_curve(2, quad_segs=1), lr1.parallel_offset(2, 'left', resolution=1))
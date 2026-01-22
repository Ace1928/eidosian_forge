import unittest
import pytest
from shapely.geometry import LinearRing, LineString
from shapely.testing import assert_geometries_equal
def test_parallel_offset_linestring(self):
    line1 = LineString([(0, 0), (10, 0)])
    left = line1.parallel_offset(5, 'left')
    assert_geometries_equal(left, LineString([(0, 5), (10, 5)]))
    right = line1.parallel_offset(5, 'right')
    assert_geometries_equal(right, LineString([(10, -5), (0, -5)]), normalize=True)
    right = line1.parallel_offset(-5, 'left')
    assert_geometries_equal(right, LineString([(10, -5), (0, -5)]), normalize=True)
    left = line1.parallel_offset(-5, 'right')
    assert_geometries_equal(left, LineString([(0, 5), (10, 5)]))
    assert_geometries_equal(line1.parallel_offset(5), right)
    line2 = LineString([(0, 0), (5, 0), (5, -5)])
    assert_geometries_equal(line2.parallel_offset(2, 'left', join_style=3), LineString([(0, 2), (5, 2), (7, 0), (7, -5)]))
    assert_geometries_equal(line2.parallel_offset(2, 'left', join_style=2), LineString([(0, 2), (7, 2), (7, -5)]))
    assert_geometries_equal(line1.offset_curve(2, quad_segs=10), line1.parallel_offset(2, 'left', resolution=10))
    assert_geometries_equal(line1.offset_curve(-2, join_style='mitre'), line1.parallel_offset(2, 'right', join_style=2))
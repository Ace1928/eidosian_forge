import unittest
import pytest
from shapely import geometry
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
def test_resolution_alias():
    point = geometry.Point(0, 0)
    result = point.buffer(1, resolution=1)
    expected = point.buffer(1, quad_segs=1)
    assert result.equals(expected)
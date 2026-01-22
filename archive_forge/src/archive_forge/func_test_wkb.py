import pickle
import struct
import unittest
from shapely import wkb, wkt
from shapely.geometry import Point
def test_wkb(self):
    p = Point(0.0, 0.0)
    wkb_big_endian = wkb.dumps(p, big_endian=True)
    wkb_little_endian = wkb.dumps(p, big_endian=False)
    assert p.equals(wkb.loads(wkb_big_endian))
    assert p.equals(wkb.loads(wkb_little_endian))
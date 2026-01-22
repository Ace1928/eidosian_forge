import pickle
import struct
import unittest
from shapely import wkb, wkt
from shapely.geometry import Point
def test_wkb_dumps_endianness(self):
    p = Point(0.5, 2.0)
    wkb_big_endian = wkb.dumps(p, big_endian=True)
    wkb_little_endian = wkb.dumps(p, big_endian=False)
    assert wkb_big_endian != wkb_little_endian
    assert wkb_big_endian[0] == 0
    assert wkb_little_endian[0] == 1
    double_size = struct.calcsize('d')
    assert wkb_big_endian[-2 * double_size:] == struct.pack('>2d', p.x, p.y)
    assert wkb_little_endian[-2 * double_size:] == struct.pack('<2d', p.x, p.y)
import binascii
import math
import struct
import sys
import pytest
from shapely import wkt
from shapely.geometry import Point
from shapely.geos import geos_version
from shapely.tests.legacy.conftest import shapely20_todo
from shapely.wkb import dump, dumps, load, loads
def test_loads_srid():
    geom = loads(hex2bin('0101000020E6100000333333333333F33F3333333333330B40'))
    assert isinstance(geom, Point)
    assert geom.coords[:] == [(1.2, 3.4)]
    result = dumps(geom)
    assert bin2hex(result) == hostorder('BIdd', '0101000000333333333333F33F3333333333330B40')
    result = dumps(geom, include_srid=True)
    assert bin2hex(result) == hostorder('BIIdd', '0101000020E6100000333333333333F33F3333333333330B40')
    result = dumps(geom, srid=27700)
    assert bin2hex(result) == hostorder('BIIdd', '0101000020346C0000333333333333F33F3333333333330B40')
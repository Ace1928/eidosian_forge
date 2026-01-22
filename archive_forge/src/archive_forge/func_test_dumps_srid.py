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
def test_dumps_srid(some_point):
    result = dumps(some_point)
    assert bin2hex(result) == hostorder('BIdd', '0101000000333333333333F33F3333333333330B40')
    result = dumps(some_point, srid=4326)
    assert bin2hex(result) == hostorder('BIIdd', '0101000020E6100000333333333333F33F3333333333330B40')
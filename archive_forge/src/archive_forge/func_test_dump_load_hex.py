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
def test_dump_load_hex(some_point, tmpdir):
    file = tmpdir.join('test.wkb')
    with open(file, 'w') as file_pointer:
        dump(some_point, file_pointer, hex=True)
    with open(file, 'r') as file_pointer:
        restored = load(file_pointer, hex=True)
    assert some_point == restored
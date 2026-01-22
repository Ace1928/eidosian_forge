import unittest
from math import pi
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
def test_rotate_empty(self):
    rls = affinity.rotate(load_wkt('LINESTRING EMPTY'), 90)
    els = load_wkt('LINESTRING EMPTY')
    assert rls.equals(els)
import unittest
from math import pi
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
def test_scale_empty(self):
    sls = affinity.scale(load_wkt('LINESTRING EMPTY'))
    els = load_wkt('LINESTRING EMPTY')
    assert sls.equals(els)
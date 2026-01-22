import unittest
from math import pi
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
def test_skew_xs_ys_array(self):
    ls = load_wkt('LINESTRING(240 400 10, 240 300 30, 300 300 20)')
    els = load_wkt('LINESTRING (253.39745962155615 417.3205080756888, 226.60254037844385 317.3205080756888, 286.60254037844385 282.67949192431126)')
    xs_ys = np.array([15.0, -30.0])
    sls = affinity.skew(ls, xs_ys[0, ...], xs_ys[1, ...])
    assert xs_ys[0] == 15.0
    assert xs_ys[1] == -30.0
    assert sls.equals_exact(els, 1e-06)
    xs_ys = np.array([pi / 12, -pi / 6])
    sls = affinity.skew(ls, xs_ys[0, ...], xs_ys[1, ...], use_radians=True)
    assert xs_ys[0] == pi / 12
    assert xs_ys[1] == -pi / 6
    assert sls.equals_exact(els, 1e-06)
import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_gdal():
    t = Affine.from_gdal(-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0)
    assert t.c == t.xoff == -237481.5
    assert t.a == 425.0
    assert t.b == 0.0
    assert t.f == t.yoff == 237536.4
    assert t.d == 0.0
    assert t.e == -425.0
    assert tuple(t) == (425.0, 0.0, -237481.5, 0.0, -425.0, 237536.4, 0, 0, 1)
    assert t.to_gdal() == (-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0)
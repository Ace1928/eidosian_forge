import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
def test_matrix():
    m = Matrix()
    with pytest.raises(AttributeError):
        m.some_inexistent_attribute
    assert m.as_tuple() == (1, 0, 0, 1, 0, 0)
    m.translate(12, 4)
    assert m.as_tuple() == (1, 0, 0, 1, 12, 4)
    m.scale(2, 7)
    assert m.as_tuple() == (2, 0, 0, 7, 12, 4)
    assert m[3] == 7
    assert m.yy == 7
    m.yy = 3
    assert m.as_tuple() == (2, 0, 0, 3, 12, 4)
    assert repr(m) == 'Matrix(2, 0, 0, 3, 12, 4)'
    assert str(m) == 'Matrix(2, 0, 0, 3, 12, 4)'
    assert m.transform_distance(1, 2) == (2, 6)
    assert m.transform_point(1, 2) == (14, 10)
    m2 = m.copy()
    assert m2 == m
    m2.invert()
    assert m2.as_tuple() == (0.5, 0, 0, 1.0 / 3, -12 / 2, -4.0 / 3)
    assert m.inverted() == m2
    assert m.as_tuple() == (2, 0, 0, 3, 12, 4)
    m2 = Matrix(*m)
    assert m2 == m
    m2.invert()
    assert m2.as_tuple() == (0.5, 0, 0, 1.0 / 3, -12 / 2, -4.0 / 3)
    assert m.inverted() == m2
    assert m.as_tuple() == (2, 0, 0, 3, 12, 4)
    m.rotate(math.pi / 2)
    assert round_tuple(m.as_tuple()) == (0, 3, -2, 0, 12, 4)
    m *= Matrix.init_rotate(math.pi)
    assert round_tuple(m.as_tuple()) == (0, -3, 2, 0, -12, -4)
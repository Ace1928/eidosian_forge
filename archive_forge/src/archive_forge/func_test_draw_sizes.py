from __future__ import annotations
import xml.etree.ElementTree
import pytest
import dask.array as da
from dask.array.svg import draw_sizes
def test_draw_sizes():
    assert draw_sizes((10, 10), size=100) == (100, 100)
    assert draw_sizes((10, 10), size=200) == (200, 200)
    assert draw_sizes((10, 5), size=100) == (100, 50)
    a, b, c = draw_sizes((1000, 100, 10))
    assert a > b
    assert b > c
    assert a < b * 5
    assert b < c * 5
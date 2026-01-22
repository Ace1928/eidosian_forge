from __future__ import annotations
import xml.etree.ElementTree
import pytest
import dask.array as da
from dask.array.svg import draw_sizes
def test_too_many_lines_fills_sides_darker():
    data = da.ones((16000, 2400, 3600), chunks=(1, 2400, 3600))
    text = data.to_svg()
    assert '8B4903' in text
    assert text.count('\n') < 300
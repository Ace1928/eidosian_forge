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
def test_solid_pattern():
    assert SolidPattern(1, 0.5, 0.25).get_rgba() == (1, 0.5, 0.25, 1)
    assert SolidPattern(1, 0.5, 0.25, 0.75).get_rgba() == (1, 0.5, 0.25, 0.75)
    surface = PDFSurface(None, 1, 1)
    context = Context(surface)
    pattern = SolidPattern(1, 0.5, 0.25)
    context.set_source(pattern)
    assert isinstance(context.get_source(), SolidPattern)
    pattern_map = cairocffi.patterns.PATTERN_TYPE_TO_CLASS
    try:
        del pattern_map[cairocffi.PATTERN_TYPE_SOLID]
        re_pattern = context.get_source()
        assert re_pattern._pointer == pattern._pointer
        assert isinstance(re_pattern, Pattern)
        assert not isinstance(re_pattern, SolidPattern)
    finally:
        pattern_map[cairocffi.PATTERN_TYPE_SOLID] = SolidPattern
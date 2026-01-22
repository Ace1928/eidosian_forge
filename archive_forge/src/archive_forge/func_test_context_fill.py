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
def test_context_fill():
    surface = ImageSurface(cairocffi.FORMAT_A8, 4, 4)
    assert surface.get_data()[:] == b'\x00' * 16
    context = Context(surface)
    context.set_source_rgba(0, 0, 0, 0.5)
    context.set_line_width(0.5)
    context.rectangle(1, 1, 2, 2)
    assert context.fill_extents() == (1, 1, 3, 3)
    assert context.stroke_extents() == (0.75, 0.75, 3.25, 3.25)
    assert context.in_fill(2, 2) is True
    assert context.in_fill(0.8, 2) is False
    assert context.in_stroke(2, 2) is False
    assert context.in_stroke(0.8, 2) is True
    path = list(context.copy_path())
    assert path
    context.fill_preserve()
    assert list(context.copy_path()) == path
    assert surface.get_data()[:] == b'\x00\x00\x00\x00\x00\x80\x80\x00\x00\x80\x80\x00\x00\x00\x00\x00'
    context.fill()
    assert list(context.copy_path()) == []
    assert surface.get_data()[:] == b'\x00\x00\x00\x00\x00\xc0\xc0\x00\x00\xc0\xc0\x00\x00\x00\x00\x00'
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
@pytest.mark.xfail(cairo_version() < 11000, reason='Cairo version too low')
def test_surface_create_for_rectangle():
    surface = ImageSurface(cairocffi.FORMAT_A8, 4, 4)
    data = surface.get_data()
    assert data[:] == b'\x00' * 16
    Context(surface.create_for_rectangle(1, 1, 2, 2)).paint()
    assert data[:] == b'\x00\x00\x00\x00\x00\xff\xff\x00\x00\xff\xff\x00\x00\x00\x00\x00'